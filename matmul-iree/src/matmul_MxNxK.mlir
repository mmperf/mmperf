func @matmul(%arg0: tensor<${M}x${K}xf32>, %arg1: tensor<${K}x${N}xf32>) -> tensor<${M}x${N}xf32>
    {
    	%0 = "mhlo.dot"(%arg0, %arg1) {name = "dot0"} : (tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>) -> tensor<${M}x${N}xf32>
    	return %0 : tensor<${M}x${N}xf32>
}
