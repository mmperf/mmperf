func @matmul(%a: memref<1020x1152xf32>, %b: memref<1152x1152xf32>, %c: memref<1020x1152xf32>) {
  linalg.matmul ins(%a, %b : memref<1020x1152xf32>, memref<1152x1152xf32>)
    outs(%c: memref<1020x1152xf32>)
  return
}