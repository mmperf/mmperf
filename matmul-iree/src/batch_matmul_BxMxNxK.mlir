func.func @batch_matmul(%arg0: tensor<1x${B}x${M}x${K}x${TYPE}>, %arg1: tensor<1x${B}x${K}x${N}x${TYPE}>) -> tensor<1x${B}x${M}x${N}x${TYPE}>
    {
	%0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [2]>
  } : (tensor<1x${B}x${M}x${K}x${TYPE}>, tensor<1x${B}x${K}x${N}x${TYPE}>) -> tensor<1x${B}x${M}x${N}x${TYPE}>
    	return %0 : tensor<1x${B}x${M}x${N}x${TYPE}>
}
