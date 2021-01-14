func @matmul(%a: memref<1920x2304xf32>, %b: memref<2304x2304xf32>, %c: memref<1920x2304xf32>) {
  linalg.matmul ins(%a, %b : memref<1920x2304xf32>, memref<2304x2304xf32>)
    outs(%c: memref<1920x2304xf32>)
  return
}