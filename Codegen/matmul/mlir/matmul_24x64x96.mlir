func @matmul(%a: memref<24x96xf32>, %b: memref<96x64xf32>, %c: memref<24x64xf32>) {
  linalg.matmul ins(%a, %b : memref<24x96xf32>, memref<96x64xf32>)
    outs(%c: memref<24x64xf32>)
  return
}