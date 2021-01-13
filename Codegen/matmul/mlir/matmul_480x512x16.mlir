func @matmul(%a: memref<480x16xf64>, %b: memref<16x512xf64>, %c: memref<480x512xf64>) {
  linalg.matmul ins(%a, %b : memref<480x16xf64>, memref<16x512xf64>)
    outs(%c: memref<480x512xf64>)
  return
}