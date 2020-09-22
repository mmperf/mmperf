#arg1 = input mlir file
#arg2 = target output file
#arg3 = mlir-hlo-opt location
#arg4 = tf-opt location
#arg5 = kernel gen opt location
#arg6 = mlir translate location

mlir_out=$1

#Original location: tensorflow/bazel-bin/tensorflow/compiler/mlir/hlo/mlir-hlo-opt
HLO_OPT=$3
HLO_OPT_FLAGS=" --transform-unranked-hlo"
HLO_OPT_FLAGS+=" --mhlo-legalize-control-flow"
HLO_OPT_FLAGS+=" --mhlo-legalize-to-std"
HLO_OPT_FLAGS+=" --allow-unregistered-dialect"


#Original location: tensorflow/bazel-bin/tensorflow/compiler/mlir/tf-opt
TF_OPT=$4
TF_OPT_FLAGS=" -canonicalize"
TF_OPT_FLAGS+=" --convert-std-to-llvm"
TF_OPT_FLAGS+=" --convert-linalg-to-loops"
TF_OPT_FLAGS+=" -lower-affine"
TF_OPT_FLAGS+=" -convert-scf-to-std"
TF_OPT_FLAGS+=" -convert-vector-to-llvm"

#Original location: tensorflow/bazel-bin/tensorflow/compiler/mlir/tools/kernel_gen/kernel-gen-opt
KERNEL_OPT=$5

KERNEL_OPT_FLAGS=" --canonicalize"
KERNEL_OPT_FLAGS+=" --shape-to-descriptors"

#Original location: llvm-project/build/bin/mlir-translate
MLIR_TRANSLATE=$6/mlir-translate

${HLO_OPT} ${HLO_OPT_FLAGS} ${mlir_out} \
 | ${TF_OPT} ${TF_OPT_FLAGS} \
 | ${KERNEL_OPT} ${KERNEL_OPT_FLAGS} \
 | ${MLIR_TRANSLATE} --mlir-to-llvmir -o $2
