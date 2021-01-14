func @matmul(%a: memref<480x16xf32>, %b: memref<16x512xf32>, %c: memref<480x512xf32>) {
  linalg.matmul ins(%a, %b : memref<480x16xf32>, memref<16x512xf32>)
    outs(%c: memref<480x512xf32>)
  return
}