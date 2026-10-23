"""Compiler-owned lowering. No author-selected tiles or thread mappings."""

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from common import write_json
from workloads import evaluate
from search_policy import current as search_policy


class TorchOps:
    def __init__(self, torch, native=False):
        self.t, self.native = torch, native

    def apply(self, node, a):
        t, op = self.t, node["op"]
        if self.native and op in {"add", "mul"}:
            # Expose the already-rounded FP16 values to the epilogue. Combined
            # with precision-cast emulation, this avoids relying on a no-op
            # half() call to survive mm+add recognition. Device validation is
            # still required; this is not a guarantee about a future compiler.
            a = [value.to(t.float32) for value in a]
        if op == "matmul":
            v = (
                t.mm(a[0], a[1], out_dtype=t.float32)
                if a[0].device.type == "cuda" and not self.native
                else a[0] @ a[1]
            )
        elif op == "add":
            v = a[0] + a[1]
        elif op == "mul":
            v = a[0] * a[1]
        elif op == "relu":
            v = t.relu(a[0])
        elif op == "transpose":
            v = a[0].T
        elif op == "cast":
            v = a[0]
        elif op == "mean":
            v = a[0].mean(-1, keepdim=True)
        elif op == "rsqrt":
            v = t.rsqrt(a[0])
        elif op == "add_scalar":
            v = a[0] + node["value"]
        elif op == "softmax":
            v = t.softmax(a[0], dim=-1)
        else:
            raise ValueError(op)
        return v.to(getattr(t, node["dtype"]))


class TorchAdapter:
    def __init__(self, program, arrays, folder, variant, trace, device="cuda"):
        import torch

        self.t, self.device = torch, device
        if device == "cuda":
            assert torch.cuda.is_available()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.inputs = [
            torch.as_tensor(a.copy(), device=device) for a in arrays.values()
        ]
        names = list(arrays)
        native = program.contract.get("matrix_adapter") == "native_low_precision"

        def fn(*args):
            return evaluate(program, TorchOps(torch, native), dict(zip(names, args)))

        self.fn = fn
        if trace:
            from torch.fx.experimental.proxy_tensor import make_fx

            graph = make_fx(fn, tracing_mode="fake")(*self.inputs)
            (folder / "01-fx.py").write_text(graph.code)
            (folder / "01-fx-graph.txt").write_text(str(graph.graph))
        start = time.perf_counter()
        if device == "cpu":
            self.compiled = fn
        else:
            mode = "max-autotune-no-cudagraphs" if variant == "tuned" else "default"
            self.compiled = torch.compile(
                fn, backend="inductor", fullgraph=True, dynamic=False, mode=mode
            )
        self.run()
        self.sync()
        self.metadata = {
            "version": torch.__version__,
            "compile_seconds": time.perf_counter() - start,
            "compile_scope": "torch.compile plus first completed invocation",
            "variant": variant,
            "submission": "Inductor wrapper; outer CUDA graphs disabled",
            "precision_adapter": (
                "FP16 inputs, native mm(out_dtype=float32), explicit FP16 "
                "result before epilogue; emulate_precision_casts enabled; "
                "reduced precision accumulation and TF32 disabled"
            ),
            "device": device,
            "cuda_runtime": torch.version.cuda,
        }
        self.metadata["semantic_capture"] = (
            "01-fx is an explicit make_fx diagnostic of the same "
            "function; torch-debug contains the actual Inductor "
            "compilation dumps"
        )
        if native:
            self.metadata["precision_adapter"] = (
                "Native FP16 matmul result; out_dtype unspecified for "
                "ordinary GEMM template eligibility. Pointwise add/mul "
                "widen their already-rounded operands then narrow the "
                "result. Precision-cast emulation enabled; FP32 "
                "accumulation and preserved boundaries require device "
                "evidence."
            )

    def set_inputs(self, arrays):
        for dst, a in zip(self.inputs, arrays.values()):
            dst.copy_(self.t.as_tensor(a, device=self.device))
        self.sync()

    def run(self):
        self.result = self.compiled(*self.inputs)

    def sync(self):
        if self.device == "cuda":
            self.t.cuda.synchronize()

    def outputs(self):
        return [x.detach().cpu().numpy() for x in self.result]


class TinyOps:
    def __init__(self, dtypes, native=False):
        self.d, self.native = dtypes, native

    def apply(self, node, a):
        op = node["op"]
        if op == "matmul":
            # dtype= requests FP32 reduction. The native path keeps FP16
            # operands and needs device evidence for multiplication precision.
            # The other path explicitly widens operands before multiplication.
            v = (
                a[0].matmul(a[1], dtype=self.d.float32)
                if self.native
                else a[0]
                .cast(self.d.float32)
                .matmul(a[1].cast(self.d.float32), dtype=self.d.float32)
            )
        elif op == "add":
            v = a[0] + a[1]
        elif op == "mul":
            v = a[0] * a[1]
        elif op == "relu":
            v = a[0].relu()
        elif op == "transpose":
            v = a[0].T
        elif op == "cast":
            v = a[0]
        elif op == "mean":
            v = a[0].mean(-1, keepdim=True)
        elif op == "rsqrt":
            v = a[0].rsqrt()
        elif op == "add_scalar":
            v = a[0] + node["value"]
        elif op == "softmax":
            v = a[0].softmax(-1)
        else:
            raise ValueError(op)
        return v.cast(getattr(self.d, node["dtype"]))


class TinyAdapter:
    def __init__(self, program, arrays, folder, variant, trace, device="CUDA"):
        from tinygrad import Tensor, dtypes
        from tinygrad.device import Device
        from tinygrad.engine.realize import (
            compile_linear,
            link_linear,
            run_linear,
            to_program_cache,
        )
        from tinygrad.uop.ops import Ops

        self.Tensor, self.device, self.dev, self.run_linear = (
            Tensor,
            device,
            Device[device],
            run_linear,
        )
        self.inputs = {k: Tensor(v, device=device).realize() for k, v in arrays.items()}
        native = program.contract.get("matrix_adapter") == "native_low_precision"
        self.result = evaluate(program, TinyOps(dtypes, native), self.inputs)
        (folder / "01-semantic.uop.txt").write_text(
            "\n\n".join(str(x.uop) for x in self.result)
        )
        start = time.perf_counter()
        linear = self.result[0].schedule_linear(*self.result[1:])
        (folder / "02-scheduled.uop.txt").write_text(str(linear))
        to_program_cache.clear()
        # Search diagnostics must not enable instrumentation in the final HCQ
        # submission program (compile_linear otherwise derives this from DEBUG).
        compiled = compile_linear(
            linear,
            beam=search_policy().get("tinygrad_beam", 2 if variant == "tuned" else 0),
            profile=False,
        )
        (folder / "03-compiled.uop.txt").write_text(str(compiled))
        selected = {
            u.arg for u in compiled.toposort(enter_calls=True) if u.op is Ops.BINARY
        }
        programs = []
        for i, prg in enumerate(to_program_cache.values()):
            info = {
                "id": i,
                "metadata": repr(prg.arg),
                "kernel_info": repr(prg.src[0].arg),
                "selected": any(
                    u.arg in selected for u in prg.src if u.op is Ops.BINARY
                ),
                "global_size": list(prg.arg.global_size),
                "local_size": list(prg.arg.local_size),
            }
            for j, u in enumerate(prg.toposort()):
                if u.op is Ops.SOURCE:
                    (folder / f"kernel-{i}-{j}.txt").write_text(str(u.arg))
                if u.op is Ops.BINARY and isinstance(u.arg, bytes):
                    suffix = (
                        "ptx"
                        if b".version " in u.arg[:1024] and b".target " in u.arg[:1024]
                        else "bin"
                    )
                    (folder / f"kernel-{i}-{j}.{suffix}").write_bytes(u.arg)
            programs.append(info)
        self.linked = link_linear(compiled)
        self.run()
        self.sync()
        self.metadata = {
            "compile_seconds": time.perf_counter() - start,
            "compile_scope": (
                "scheduling, compilation/search, UOp and candidate-code "
                "artifact export, link, first execution"
            ),
            "programs": programs,
            "variant": variant,
            "device": device,
            "hcq_profile_instrumented": False,
            "precision_adapter": (
                "Stored FP16 operands explicitly widened before "
                "multiplication; FP32 reduction; FP16 result before "
                "epilogue. No TF32 requested."
            ),
            "submission": "precompiled linked linear program; run_linear(jit=True)",
        }
        write_json(folder / "programs.json", programs)
        if native:
            self.metadata["precision_adapter"] = (
                "Native FP16 operands, requested FP32 reduction, FP16 "
                "node results. Scalar fallback may round products; "
                "multiplication probe and selected matrix-instruction "
                "inspection are mandatory."
            )

    def set_inputs(self, arrays):
        for k, a in arrays.items():
            self.inputs[k].assign(self.Tensor(a, device=self.device)).realize()
        self.sync()

    def run(self):
        self.run_linear(self.linked, jit=True, wait=False, update_stats=False)

    def sync(self):
        self.dev.synchronize()

    def outputs(self):
        return [v.numpy() for v in self.result]


class MaxOps:
    def __init__(self, ops, dtype, device):
        self.ops, self.dtype, self.device = ops, dtype, device

    def apply(self, node, a):
        ops, kind = self.ops, node["op"]
        if kind == "matmul":
            v = ops.matmul(*a)
        elif kind == "add":
            v = a[0] + a[1]
        elif kind == "mul":
            v = a[0] * a[1]
        elif kind == "relu":
            v = ops.relu(a[0])
        elif kind == "transpose":
            v = ops.transpose(a[0], 0, 1)
        elif kind == "cast":
            v = a[0]
        elif kind == "mean":
            v = ops.mean(a[0], axis=-1)
        elif kind == "rsqrt":
            v = ops.rsqrt(a[0])
        elif kind == "add_scalar":
            v = a[0] + ops.constant(
                node["value"], getattr(self.dtype, node["dtype"]), self.device
            )
        elif kind == "softmax":
            v = ops.softmax(a[0], axis=-1)
        else:
            raise ValueError(kind)
        return ops.cast(v, getattr(self.dtype, node["dtype"]))


class MaxAdapter:
    def __init__(self, program, arrays, folder, variant, trace, cpu=False):
        import importlib.metadata
        from max.driver import Accelerator, Buffer, CPU
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops

        self.Buffer, self.CPU, self.device = (
            Buffer,
            CPU,
            CPU() if cpu else Accelerator(0),
        )
        ref = DeviceRef.from_device(self.device)
        with Graph(
            program.name,
            input_types=[
                TensorType(DType.float16, a.shape, ref) for a in arrays.values()
            ],
        ) as graph:
            outputs = evaluate(
                program,
                MaxOps(ops, DType, ref),
                dict(zip(arrays, [v.tensor for v in graph.inputs])),
            )
            graph.output(*outputs)
        (folder / "01-semantic.mlir").write_text(str(graph))
        start = time.perf_counter()
        self.session = InferenceSession(devices=[self.device])
        compiled = self.session.compile(graph, tile_based_fusion=False)
        self.model = self.session.init(compiled)
        self.set_inputs(arrays)
        self.run()
        self.sync()
        seconds = time.perf_counter() - start
        compiled.export_mef(str(folder / "executable.mef"))
        self.metadata = {
            "version": importlib.metadata.version("max"),
            "compile_seconds": seconds,
            "compile_scope": "session, compile, init, upload and first execution; excludes MEF export",
            "device": str(self.device),
            "precision_adapter": "FP16 native matmul; arithmetic policy must also be checked in emitted code",
            "submission": "MAX model.execute; explicit device synchronize",
            "tile_based_fusion": False,
        }

    def set_inputs(self, arrays):
        self.inputs = [
            self.Buffer.from_numpy(a).to(self.device) for a in arrays.values()
        ]
        self.sync()

    def run(self):
        self.result = self.model.execute(*self.inputs)

    def sync(self):
        self.device.synchronize()

    def outputs(self):
        return [x.to(self.CPU()).to_numpy() for x in self.result]


class TVMOps:
    def __init__(self, relax, bb):
        self.r, self.bb = relax, bb

    def apply(self, node, a):
        r, kind = self.r, node["op"]
        if kind == "matmul":
            v = r.op.matmul(*a, out_dtype="float32")
        elif kind == "add":
            v = r.op.add(*a)
        elif kind == "mul":
            v = r.op.multiply(*a)
        elif kind == "relu":
            v = r.op.nn.relu(a[0])
        elif kind == "transpose":
            v = r.op.permute_dims(a[0], axes=[1, 0])
        elif kind == "cast":
            v = a[0]
        elif kind == "mean":
            v = r.op.mean(a[0], axis=[-1], keepdims=True)
        elif kind == "rsqrt":
            v = r.op.rsqrt(a[0])
        elif kind == "add_scalar":
            v = r.op.add(a[0], r.const(node["value"], node["dtype"]))
        elif kind == "softmax":
            v = r.op.nn.softmax(a[0], axis=-1)
        else:
            raise ValueError(kind)
        return self.bb.emit(r.op.astype(v, node["dtype"]))


class TVMAdapter:
    def __init__(
        self, program, arrays, folder, variant, trace, cpu=False, schedule_only=False
    ):
        import tvm
        from tvm import relax
        from tvm.s_tir import dlight

        self.tvm, self.device = tvm, tvm.cpu() if cpu else tvm.cuda(0)
        target = tvm.target.Target(
            "llvm" if cpu else {"kind": "cuda", "arch": "sm_90"}, host="llvm"
        )
        bb = relax.BlockBuilder()
        variables = [
            relax.Var(k, relax.TensorType(v.shape, "float16"))
            for k, v in arrays.items()
        ]
        with bb.function("main", variables):
            with bb.dataflow():
                ys = evaluate(program, TVMOps(relax, bb), dict(zip(arrays, variables)))
                out = bb.emit_output(relax.Tuple(list(ys)))
            bb.emit_func_output(out)
        mod = bb.get()
        (folder / "01-relax.py").write_text(mod.script())
        start = time.perf_counter()
        mod = relax.transform.LegalizeOps()(mod)
        (folder / "02-legalized.py").write_text(mod.script())
        mod = tvm.transform.Sequential(
            [
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ]
        )(mod)
        (folder / "03-fused.py").write_text(mod.script())
        traces, replay = [], []
        if not cpu:
            with target:
                for gv, func in list(mod.functions_items()):
                    if not isinstance(func, tvm.tirx.PrimFunc):
                        continue
                    for rule in (
                        dlight.gpu.Matmul(),
                        dlight.gpu.GEMV(),
                        dlight.gpu.Reduction(),
                        dlight.gpu.GeneralReduction(),
                        dlight.gpu.Fallback(),
                    ):
                        sch = rule.apply(func, target, False)
                        if sch is None:
                            continue
                        if isinstance(sch, list):
                            if len(sch) != 1:
                                raise RuntimeError(
                                    "DLight returned ambiguous schedules"
                                )
                            sch = sch[0]
                        traces.append(
                            f"# {gv.name_hint}: {type(rule).__name__}\n"
                            + "\n".join(sch.trace.as_python())
                        )
                        trace_path = folder / "schedules" / f"{gv.name_hint}.py"
                        trace_path.parent.mkdir(exist_ok=True)
                        trace_path.write_text("\n".join(sch.trace.as_python()) + "\n")
                        replay_sch = tvm.s_tir.Schedule(func)
                        exec(
                            compile(trace_path.read_text(), str(trace_path), "exec"),
                            {"sch": replay_sch, "T": tvm.script.tirx},
                        )
                        tvm.ir.assert_structural_equal(sch.mod, replay_sch.mod)
                        replay.append(
                            {
                                "function": gv.name_hint,
                                "passed": True,
                                "replayed_file": str(trace_path.relative_to(folder)),
                            }
                        )
                        mod[gv] = sch.mod["main"].with_attr("tirx.is_scheduled", True)
                        break
                    else:
                        raise RuntimeError(f"No GPU schedule for {gv.name_hint}")
        (folder / "04-schedule-trace.py").write_text("\n\n".join(traces))
        write_json(folder / "schedule-replay.json", replay)
        (folder / "05-scheduled.py").write_text(mod.script())
        if schedule_only:
            self.metadata = {
                "scope": "CUDA-target schedule construction and replay only; no CUDA code generation or execution",
                "schedule_replay": replay,
            }
            return

        @tvm.instrument.pass_instrument
        class CapturePasses:
            def __init__(self):
                self.count = 0
                self.records = []

            def run_after_pass(self, module, info):
                keys = (
                    "Memory",
                    "Buffer",
                    "Thread",
                    "Tensor",
                    "Storage",
                    "HostDevice",
                    "Launch",
                    "Packed",
                    "Pipeline",
                )
                if self.count >= 80 or not any(key in info.name for key in keys):
                    return
                self.count += 1
                name = f"passes/{self.count:03d}-{info.name}.py"
                (folder / "passes").mkdir(exist_ok=True)
                (folder / name).write_text(module.script())
                self.records.append({"pass": info.name, "file": name})

        capture_passes = CapturePasses()
        # Retain exactly what the normal compiler callback returns. Runtime
        # modules need not expose a writable cubin format after construction.
        import tvm_ffi

        original_cuda_compile = (
            tvm_ffi.get_global_func("tvm_callback_cuda_compile")
            if trace and not cpu
            else None
        )
        compiled_artifacts = []

        def capture_cuda_compile(code):
            binary = original_cuda_compile(code)
            index = len(compiled_artifacts)
            stem = folder / f"cuda-compile-{index}"
            stem.with_suffix(".cu").write_text(str(code))
            stem.with_suffix(".cubin").write_bytes(bytes(binary))
            compiled_artifacts.append(
                {
                    "source": stem.with_suffix(".cu").name,
                    "binary": stem.with_suffix(".cubin").name,
                    "provenance": "Unmodified result of native tvm_callback_cuda_compile (NVRTC default)",
                }
            )
            return binary

        if original_cuda_compile is not None:
            tvm_ffi.register_global_func(
                "tvm_callback_cuda_compile", capture_cuda_compile, override=True
            )
        try:
            with tvm.transform.PassContext(
                instruments=[capture_passes] if trace else []
            ):
                executable = tvm.compile(mod, target=target)
        finally:
            if original_cuda_compile is not None:
                tvm_ffi.register_global_func(
                    "tvm_callback_cuda_compile", original_cuda_compile, override=True
                )
                write_json(folder / "cuda-compile-provenance.json", compiled_artifacts)
        write_json(folder / "pass-index.json", capture_passes.records)
        seen = set()

        def capture(module):
            if module in seen:
                return
            seen.add(module)
            try:
                code = module.inspect_source()
                if code:
                    (folder / f"06-module-{len(seen)}.txt").write_text(code)
            except Exception as e:
                (folder / f"06-module-{len(seen)}-unavailable.txt").write_text(str(e))
            if trace:
                try:
                    formats = list(module.get_write_formats())
                    write_json(
                        folder / f"06-module-{len(seen)}-formats.json",
                        {"kind": str(module.kind), "formats": formats},
                    )
                    for fmt in ("cubin", "ptx"):
                        if fmt in formats:
                            module.write_to_file(
                                str(folder / f"06-module-{len(seen)}.{fmt}"), fmt
                            )
                except Exception as e:
                    (
                        folder / f"06-module-{len(seen)}-export-unavailable.txt"
                    ).write_text(str(e))
            for child in module.imports:
                capture(child)

        capture(executable.mod)
        self.vm = relax.VirtualMachine(executable, self.device)
        self.set_inputs(arrays)
        self.run()
        self.sync()
        self.metadata = {
            "version": tvm.__version__,
            "compile_seconds": time.perf_counter() - start,
            "compile_scope": (
                "legalization, fusion, DLight, replay verification, "
                "codegen, VM initialization and first invocation"
            ),
            "target": str(target),
            "selection": "DLight default rules; no MetaSchedule search",
            "precision_adapter": "FP16 operands, FP32 matmul result followed by explicit FP16 cast",
            "submission": "Relax VM invocation",
        }

    def set_inputs(self, arrays):
        self.inputs = [self.tvm.runtime.tensor(a, self.device) for a in arrays.values()]
        self.sync()

    def run(self):
        self.result = self.vm["main"](*self.inputs)

    def sync(self):
        self.device.sync()

    def outputs(self):
        return [x.numpy() for x in self.result]


class LuminalAdapter:
    """Rust owns device execution; JSON commands stay outside the timed region."""

    def __init__(self, program, arrays, folder, variant, trace, fixture_root):
        binary = os.environ.get("LUMINAL_CASE_BINARY", "/opt/cuda-case-luminal")
        self.process = subprocess.Popen(
            [
                binary,
                str(fixture_root),
                str(folder),
                variant,
                "trace" if trace else "bench",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.folder = folder
        self.metadata = self.receive()
        self.metadata["submission"] = (
            "Luminal native CUDA graph execution; synchronization inside runtime"
        )

    def receive(self):
        for line in self.process.stdout:
            if line.startswith("CASE_JSON:"):
                response = json.loads(line[len("CASE_JSON:") :])
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response
            print(line.rstrip(), flush=True)
        raise RuntimeError(f"Luminal worker stopped ({self.process.poll()})")

    def request(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.receive()

    def validate_fixture(self, directory):
        reply = self.request({"op": "validate", "directory": str(directory)})
        return [
            np.fromfile(item["path"], dtype="<f2").reshape(item["shape"])
            for item in reply["outputs"]
        ]

    def benchmark(self, blocks):
        return self.request({"op": "benchmark", "blocks": blocks})

    def profile(self):
        return self.request({"op": "profile"})

    def close(self):
        if self.process.poll() is None:
            self.request({"op": "stop"})
            self.process.wait(timeout=10)
