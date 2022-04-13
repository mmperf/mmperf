func @matmul(%arg0: tensor<${M}x${K}xf16>, %arg1: tensor<${K}x${N}xf16>) -> tensor<${M}x${N}xf16>
    {
    	%0 = "mhlo.dot"(%arg0, %arg1) {name = "dot0"} : (tensor<${M}x${K}xf16>, tensor<${K}x${N}xf16>) -> tensor<${M}x${N}xf16>
    	return %0 : tensor<${M}x${N}xf16>
}
