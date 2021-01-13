func @matmul(%a: memref<24x512xf64>, %b: memref<512x64xf64>, %c: memref<24x64xf64>) {
  linalg.matmul ins(%a, %b : memref<24x512xf64>, memref<512x64xf64>)
    outs(%c: memref<24x64xf64>)
  return
}