func.func @matmul(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>, %c: tensor<256x256xf32>)
  -> tensor<256x256xf32>
{
  %f0 = arith.constant 0.0 : f32
  %f1 = linalg.fill ins(%f0 : f32) outs(%c : tensor<256x256xf32>) -> tensor<256x256xf32>
  %d = linalg.matmul ins(%a, %b : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%f1: tensor<256x256xf32>) -> tensor<256x256xf32>
  %e = linalg.matmul ins(%a, %d : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%f1: tensor<256x256xf32>) -> tensor<256x256xf32>
  return %e: tensor<256x256xf32>
}
