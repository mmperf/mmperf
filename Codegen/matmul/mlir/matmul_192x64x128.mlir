func @matmul(%a: memref<192x128xf32>, %b: memref<128x64xf32>, %c: memref<192x64xf32>) {
  linalg.matmul ins(%a, %b : memref<192x128xf32>, memref<128x64xf32>)
    outs(%c: memref<192x64xf32>)
  return
}