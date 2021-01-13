func @matmul(%a: memref<480x256xf64>, %b: memref<256x512xf64>, %c: memref<480x512xf64>) {
  linalg.matmul ins(%a, %b : memref<480x256xf64>, memref<256x512xf64>)
    outs(%c: memref<480x512xf64>)
  return
}