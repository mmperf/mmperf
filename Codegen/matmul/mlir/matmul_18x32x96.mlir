func @matmul(%a: memref<18x96xf32>, %b: memref<96x32xf32>, %c: memref<18x32xf32>) {
  linalg.matmul ins(%a, %b : memref<18x96xf32>, memref<96x32xf32>)
    outs(%c: memref<18x32xf32>)
  return
}