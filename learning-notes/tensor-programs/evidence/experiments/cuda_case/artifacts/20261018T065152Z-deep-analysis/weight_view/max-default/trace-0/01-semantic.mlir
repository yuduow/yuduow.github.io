mo.graph @weight_view(%arg0: !mo.tensor<[1024, 768], f16, gpu:0>, %arg1: !mo.tensor<[1024, 768], f16, gpu:0>, %arg2: !mo.tensor<[1024], f16, gpu:0>) -> !mo.tensor<[1024, 1024], f16, gpu:0> attributes {_kernel_library_paths = [], argument_names = ["input0", "input1", "input2"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %1 = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %2 = rmo.mo.transpose(%arg1, %1) : (!mo.tensor<[1024, 768], f16, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[768, 1024], f16, gpu:0>
  %3 = rmo.matmul(%arg0, %2) : (!mo.tensor<[1024, 768], f16, gpu:0>, !mo.tensor<[768, 1024], f16, gpu:0>) -> !mo.tensor<[1024, 1024], f16, gpu:0>
  %4 = rmo.add(%3, %arg2) : (!mo.tensor<[1024, 1024], f16, gpu:0>, !mo.tensor<[1024], f16, gpu:0>) -> !mo.tensor<[1024, 1024], f16, gpu:0>
  mo.output %4 : !mo.tensor<[1024, 1024], f16, gpu:0>
}