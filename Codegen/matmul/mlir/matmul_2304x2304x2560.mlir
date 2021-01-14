func @matmul(%a: memref<2304x2560xf32>, %b: memref<2560x2304xf32>, %c: memref<2304x2304xf32>) {
  linalg.matmul ins(%a, %b : memref<2304x2560xf32>, memref<2560x2304xf32>)
    outs(%c: memref<2304x2304xf32>)
  return
}