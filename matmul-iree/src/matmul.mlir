func @matmul() -> tensor<2x2xf32>
    {
    %arg0 = constant dense<[[1.], [2.]]> : tensor<2x1xf32>
    %arg1 = constant dense<[[3., 4.]]> : tensor<1x2xf32>
    %0 = "mhlo.dot"(%arg0, %arg1) {name = "dot0"} : (tensor<2x1xf32>, tensor<1x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
}
