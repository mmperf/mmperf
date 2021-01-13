func @matmul(%a: memref<192x128xf64>, %b: memref<128x128xf64>, %c: memref<192x128xf64>) {
  linalg.matmul ins(%a, %b : memref<192x128xf64>, memref<128x128xf64>)
    outs(%c: memref<192x128xf64>)
  return
}