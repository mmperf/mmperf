func @matmul(%a: memref<2304x2560xf64>, %b: memref<2560x2304xf64>, %c: memref<2304x2304xf64>) {
  linalg.matmul ins(%a, %b : memref<2304x2560xf64>, memref<2560x2304xf64>)
    outs(%c: memref<2304x2304xf64>)
  return
}