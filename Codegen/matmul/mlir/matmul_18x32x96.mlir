func @matmul(%a: memref<18x96xf64>, %b: memref<96x32xf64>, %c: memref<18x32xf64>) {
  linalg.matmul ins(%a, %b : memref<18x96xf64>, memref<96x32xf64>)
    outs(%c: memref<18x32xf64>)
  return
}