func.func @matmul(%arg0: tensor<${M}x${K}x${TYPE}>, %arg1: tensor<${K}x${N}x${TYPE}>) -> tensor<${M}x${N}x${TYPE}>
    {
    	%0 = "mhlo.dot"(%arg0, %arg1) {name = "dot0"} : (tensor<${M}x${K}x${TYPE}>, tensor<${K}x${N}x${TYPE}>) -> tensor<${M}x${N}x${TYPE}>
    	return %0 : tensor<${M}x${N}x${TYPE}>
}
