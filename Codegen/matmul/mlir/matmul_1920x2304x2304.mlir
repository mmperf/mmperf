func @matmul(%a: memref<1920x2304xf64>, %b: memref<2304x2304xf64>, %c: memref<1920x2304xf64>) {
  linalg.matmul ins(%a, %b : memref<1920x2304xf64>, memref<2304x2304xf64>)
    outs(%c: memref<1920x2304xf64>)
  return
}