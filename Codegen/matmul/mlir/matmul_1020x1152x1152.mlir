func @matmul(%a: memref<1020x1152xf64>, %b: memref<1152x1152xf64>, %c: memref<1020x1152xf64>) {
  linalg.matmul ins(%a, %b : memref<1020x1152xf64>, memref<1152x1152xf64>)
    outs(%c: memref<1020x1152xf64>)
  return
}