func @matmul(%a: memref<1024x1024xf32>, %b: memref<1024x1024xf32>, %c: memref<1024x1024xf32>) {
  linalg.matmul ins(%a, %b : memref<1024x1024xf32>, memref<1024x1024xf32>)
    outs(%c: memref<1024x1024xf32>)
  return
}