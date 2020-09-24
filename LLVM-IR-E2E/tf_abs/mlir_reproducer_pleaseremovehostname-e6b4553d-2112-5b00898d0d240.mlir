// configuration: -pass-pipeline='func(xla-legalize-tf{allow-partial-conversion=false device-type=INVALID_DEVICE_TYPE legalize-chlo=true use-tf2xla-fallback=false}, materialize-broadcast, unfuse-batch-norm), unknown<mlir::mhlo::{anonymous}::HloLegalizeToLhlo>{results-escape-function=true}, func(buffer-placement, unknown<{anonymous}::CopyRemovalPass>), shape-to-descriptors, canonicalize, func(unknown<mlir::{anonymous}::LhloLegalizeToLinalgPass>, unknown<mlir::lmhlo::{anonymous}::LhloFuseLinalgPass>{tile-sizes= use-parallel-loops=true}, convert-linalg-to-parallel-loops, canonicalize, cse, unknown<xla::mlir_gpu::{anonymous}::FuseInnerParallelLoopsPass>, cse, unknown<xla::mlir_gpu::{anonymous}::StoreForwardingPass>, unknown<xla::mlir_gpu::{anonymous}::DeadTempBufferRemovalPass>, canonicalize, cse, unknown<xla::mlir_gpu::{anonymous}::MapParallelLoopsPass>), convert-parallel-loops-to-gpu, func(canonicalize, cse, unknown<mlir::mhlo::{anonymous}::LegalizeTrigonometricToApproximationPass>, unknown<xla::mlir_gpu::{anonymous}::MoveScalarComputationsIntoGpuLaunchPass>), gpu-kernel-outlining, func(unknown<xla::mlir_gpu::{anonymous}::RewriteKernelSignaturePass>), lower-affine, convert-scf-to-std'
// note: verifyPasses=true


module {
  func @abs(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %0 = "tf.Abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }
}