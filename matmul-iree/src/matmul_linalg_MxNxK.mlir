func.func @matmul(%a: tensor<${M}x${K}xf32>, %b: tensor<${K}x${N}xf32>)
  -> tensor<${M}x${N}xf32>
{
  %c = arith.constant dense<0.0> : tensor<${M}x${N}xf32>
  %d = linalg.matmul ins(%a, %b : tensor<${M}x${K}xf32>, tensor<${K}x${N}xf32>)
    outs(%c: tensor<${M}x${N}xf32>) -> tensor<${M}x${N}xf32>
  return %d: tensor<${M}x${N}xf32>
}
