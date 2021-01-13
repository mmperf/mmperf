func @matmul(%a: memref<1024x1024xf64>, %b: memref<1024x1024xf64>, %c: memref<1024x1024xf64>) {
  linalg.matmul ins(%a, %b : memref<1024x1024xf64>, memref<1024x1024xf64>)
    outs(%c: memref<1024x1024xf64>)
  return
}