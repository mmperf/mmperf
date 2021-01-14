func @matmul(%a: memref<48x128xf32>, %b: memref<128x64xf32>, %c: memref<48x64xf32>) {
  linalg.matmul ins(%a, %b : memref<48x128xf32>, memref<128x64xf32>)
    outs(%c: memref<48x64xf32>)
  return
}