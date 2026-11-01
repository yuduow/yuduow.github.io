  mo.graph @weight_view(%arg0: !mo.tensor<[1024, 768], f16, gpu:0> loc(unknown), %arg1: !mo.tensor<[1024, 768], f16, gpu:0> loc(unknown), %arg2: !mo.tensor<[1024], f16, gpu:0> loc(unknown)) -> !mo.tensor<[1024, 1024], f16, gpu:0> attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2"], result_names = ["output0"]} {
    %0 = mo.rebind(%arg1) : !mo.tensor<[1024, 768], f16, gpu:0> -> !mo.tensor<[1024, 768], f16, gpu:0, {layout = #mo.layout<NK>}> loc(#loc)
    %1 = mogg.routine (%arg0, %0, %arg2 : !mo.tensor<[1024, 768], f16, gpu:0>, !mo.tensor<[1024, 768], f16, gpu:0, {layout = #mo.layout<NK>}>, !mo.tensor<[1024], f16, gpu:0>) -> !mo.tensor<[1024, 1024], f16, gpu:0>
      inputs (%arg3, %arg4, %arg5)
      outputs (%arg6)
      context (%arg7) attributes {device = #M.device_ref<"gpu", 0>, fusionInfo = #matmul_static_broadcast_to_add, hasReadEffect = false, hasWriteEffect = false} {
      mogg.call @__matmul_static_broadcast_to_add %arg3, %arg4, %arg5, %arg6, %arg7 : !mogg.pointer, !mogg.pointer, !mogg.pointer, !mogg.pointer, !mogg.context loc(#loc)
    } loc(#loc)
    mo.output %1 : !mo.tensor<[1024, 1024], f16, gpu:0> loc(#loc)
  } loc(#loc)
