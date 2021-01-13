func @matmul(%a: memref<192x128xf64>, %b: memref<128x64xf64>, %c: memref<192x64xf64>) {
  linalg.matmul ins(%a, %b : memref<192x128xf64>, memref<128x64xf64>)
    outs(%c: memref<192x64xf64>)
  return
}