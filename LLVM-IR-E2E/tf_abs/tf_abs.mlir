func @abs(%arg0: tensor<*xf32>) -> tensor<*xf32> {
 %0 = "tf.Abs"(%arg0) { }
 : (tensor<*xf32>) -> tensor<*xf32>
 return %0 : tensor<*xf32>
}
