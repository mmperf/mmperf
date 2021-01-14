func @matmul(%a: memref<480x256xf32>, %b: memref<256x512xf32>, %c: memref<480x512xf32>) {
  linalg.matmul ins(%a, %b : memref<480x256xf32>, memref<256x512xf32>)
    outs(%c: memref<480x512xf32>)
  return
}