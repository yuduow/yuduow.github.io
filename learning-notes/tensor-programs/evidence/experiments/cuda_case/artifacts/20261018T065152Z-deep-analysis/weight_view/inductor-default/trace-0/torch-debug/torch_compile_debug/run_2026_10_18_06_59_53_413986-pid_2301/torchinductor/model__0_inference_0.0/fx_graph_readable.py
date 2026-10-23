class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f16[1024, 768]", arg1_1: "f16[1024, 768]", arg2_1: "f16[1024]"):
        # File: /workspace/experiments/cuda_case/adapters.py:27 in apply, code: elif op == "transpose": v = a[0].T
        permute: "f16[768, 1024]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None

        # File: /workspace/experiments/cuda_case/adapters.py:23 in apply, code: v = t.mm(a[0], a[1], out_dtype=t.float32) if a[0].device.type=='cuda' and not self.native else a[0] @ a[1]
        mm: "f16[1024, 1024]" = torch.ops.aten.mm.default(arg1_1, permute);  arg1_1 = permute = None

        # File: /workspace/experiments/cuda_case/adapters.py:21 in apply, code: a = [value.to(t.float32) for value in a]
        convert_element_type_2: "f32[1024, 1024]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_3: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None

        # File: /workspace/experiments/cuda_case/adapters.py:24 in apply, code: elif op == "add": v = a[0] + a[1]
        add: "f32[1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_2, convert_element_type_3);  convert_element_type_2 = convert_element_type_3 = None

        # File: /workspace/experiments/cuda_case/adapters.py:34 in apply, code: return v.to(getattr(t, node["dtype"]))
        convert_element_type_4: "f16[1024, 1024]" = torch.ops.prims.convert_element_type.default(add, torch.float16);  add = None
        return (convert_element_type_4,)
