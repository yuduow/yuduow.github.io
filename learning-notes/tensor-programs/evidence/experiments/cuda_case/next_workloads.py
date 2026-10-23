"Second campaign: ten controlled FP16 programs, with native-matrix eligibility gates."

from workloads import Program, SHAPES

# The post follows `make_program("weight_view")`: transpose, MatMul, then bias.
# Shapes below use (M, K, N): input rows, input features, output features.
SHAPES["matrix"] = (1024, 768, 1024)  # aligned but non-square weights
SHAPES["matrix_inspect"] = (8, 16, 24)  # CPU shape/orientation check only
QUESTIONS = {
    "gemm": (
        "x @ w",
        "Can native FP16 operands reach FP32-accumulating matrix instructions?",
    ),
    "bias": (
        "x @ w + b",
        "Where is the FP16 projection rounded and the row-broadcast bias placed?",
    ),
    "weight_view": (
        "x @ wt.T + b",
        "Does a transposed weight become indexing, a library descriptor, or a copy?",
    ),
    "input_scale": (
        "(x * s) @ w",
        "Is a column scale materialized before GEMM or absorbed into its input loads?",
    ),
    "output_scale": (
        "(x @ w) * s",
        "Does a column scale enter the matrix epilogue, respecting the projection rounding?",
    ),
    "residual": (
        "x @ w + residual",
        "How does a full-size second input differ from broadcast bias?",
    ),
    "public_projection": (
        "p = x @ w; return p + b, p",
        "Does exposing the projection change storage, fusion, or recomputation?",
    ),
    "relu": (
        "relu(x @ w + b)",
        "Which activation representation and epilogue implementations remain available?",
    ),
    "projection_join": (
        "(x @ w) + (x @ v)",
        "How are two independent contractions and their join partitioned?",
    ),
    "projection_chain": (
        "(x @ w) @ v",
        "When does a contraction consumer force storage and a second launch?",
    ),
}

PAIRS = {
    "gemm": None,
    "bias": "gemm",
    "weight_view": "bias",
    "input_scale": "gemm",
    "output_scale": "gemm",
    "residual": "bias",
    "public_projection": "bias",
    "relu": "bias",
    "projection_join": "gemm",
    "projection_chain": "gemm",
}


def make_program(name, shape_name="matrix"):
    m, k, n = SHAPES[shape_name]
    inputs, nodes = {}, []

    def inp(label, shape):
        inputs[label] = {"shape": list(shape), "dtype": "float16"}
        return label

    def op(kind, *args, **attrs):
        label = f"v{len(nodes)}"
        nodes.append(
            {"id": label, "op": kind, "args": list(args), "dtype": "float16", **attrs}
        )
        return label

    x = inp("x", (m, k))
    if name == "input_scale":
        x = op("mul", x, inp("s", (k,)))
    w = (
        op("transpose", inp("wt", (n, k)), axes=[1, 0])
        if name == "weight_view"
        else inp("w", (k, n))
    )
    p = op("matmul", x, w, accumulation="float32")
    y = p
    if name in {"bias", "weight_view", "public_projection", "relu"}:
        y = op("add", p, inp("b", (n,)))
    if name == "relu":
        y = op("relu", y)
    if name == "residual":
        y = op("add", p, inp("residual", (m, n)))
    if name == "output_scale":
        y = op("mul", p, inp("s", (n,)))
    if name == "projection_join":
        y = op("add", p, op("matmul", x, inp("v", (k, n)), accumulation="float32"))
    if name == "projection_chain":
        y = op("matmul", p, inp("v", (n, n)), accumulation="float32")
    return Program(
        name,
        shape_name,
        (m, k, n),
        *QUESTIONS[name],
        inputs,
        nodes,
        [y, p] if name == "public_projection" else [y],
        {
            "campaign": "native-matrix-v2",
            "matrix_adapter": "native_low_precision",
            "paired_with": PAIRS[name],
            "input_output_storage": "float16",
            "matmul_accumulation_requested": "float32",
            "rounding": (
                "Each declared node rounds to FP16, including GEMM before "
                "its epilogue. FP16 products rounded before accumulation "
                "are not permitted."
            ),
            "reductions": (
                "FP32 accumulation; reassociation/FMA permitted; TF32 and "
                "reduced-precision reduction disabled."
            ),
            "reference": (
                "Independent FP64 operations on exact stored FP16 inputs, "
                "rounded at declared node boundaries."
            ),
            "atol": 0.0002,
            "rtol": 0.002,
            "admission": (
                "Both independent validation and inspection of the "
                "selected matrix arithmetic are required. A failed native "
                "path is recorded, never silently replaced with widened "
                "operands."
            ),
        },
    )
