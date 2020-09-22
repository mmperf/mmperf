#arg1 = input mlir file
#arg2 = target output file
#arg3 = mlir-hlo-opt location
#arg4 = tf-opt location
#arg5 = kernel gen opt location
#arg6 = mlir translate location

mlir_out=$1

#Original location: tensorflow/bazel-bin/tensorflow/compiler/mlir/hlo/mlir-hlo-opt
HLO_OPT=$3
HLO_OPT_FLAGS_1=" --transform-unranked-hlo"
HLO_OPT_FLAGS_1+=" --canonicalize"
HLO_OPT_FLAGS_2=" --lhlo-legalize-to-linalg"

#Original location: tensorflow/bazel-bin/tensorflow/compiler/mlir/tf-opt
TF_OPT=$4
TF_OPT_FLAGS_1=" --xla-legalize-tf"
TF_OPT_FLAGS_2=" --convert-scf-to-std"
TF_OPT_FLAGS_2+=" --convert-std-to-llvm"

#Original location: tensorflow/bazel-bin/tensorflow/compiler/mlir/tools/kernel_gen/kernel-gen-opt
KERNEL_OPT=$5

KERNEL_OPT_FLAGS=" --shape-to-descriptors"
KERNEL_OPT_FLAGS+=" --bufferize"
KERNEL_OPT_FLAGS+=" --canonicalize"

#Original location: llvm-project/build/bin/mlir-translate
MLIR_TRANSLATE=$6

${TF_OPT} ${TF_OPT_FLAGS_1} ${mlir_out} \
 | ${HLO_OPT} ${HLO_OPT_FLAGS_1} \
 | ${KERNEL_OPT} ${KERNEL_OPT_FLAGS} \
 | ${HLO_OPT} ${HLO_OPT_FLAGS_2} \
 | ${TF_OPT} ${TF_OPT_FLAGS_2} \
 | ${MLIR_TRANSLATE} --mlir-to-llvmir -o $2
