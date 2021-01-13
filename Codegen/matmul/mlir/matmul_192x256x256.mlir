func @matmul(%a: memref<192x256xf64>, %b: memref<256x256xf64>, %c: memref<192x256xf64>) {
  linalg.matmul ins(%a, %b : memref<192x256xf64>, memref<256x256xf64>)
    outs(%c: memref<192x256xf64>)
  return
}