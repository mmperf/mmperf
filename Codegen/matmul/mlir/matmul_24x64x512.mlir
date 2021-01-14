func @matmul(%a: memref<24x512xf32>, %b: memref<512x64xf32>, %c: memref<24x64xf32>) {
  linalg.matmul ins(%a, %b : memref<24x512xf32>, memref<512x64xf32>)
    outs(%c: memref<24x64xf32>)
  return
}