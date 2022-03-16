#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Error.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

#include <fstream>
#include <string>
#include <unistd.h>
#include <iostream>

#include "flatbuffers/flatbuffers.h"
#include "compile_options_generated.h"

using namespace mlir;
using namespace mlir::linalg;

using llvm::Error;
using llvm::Expected;
using llvm::StringError;
using llvm::Twine;

namespace mlir {

namespace {
namespace cl = llvm::cl;
struct Options {
  cl::opt<std::string> inputFile{cl::Positional,
    cl::desc("the input .mlir file"),
    cl::init("")};

  // CPU info
  cl::opt<std::string> targetCPU{"target-cpu", cl::Required,
    cl::desc("Target CPU for codegen"),
    cl::init("skylake-avx512")};
  cl::opt<std::string> vectorWidth{"vector-width", cl::Required,
    cl::desc("Target vector width for codegen"),
    cl::init("512")};

  // Codegen info
  cl::opt<std::string> compileOptions{"compile-options", cl::Required,
    cl::desc("Flatbuffer file describing compile options configurations"),
    cl::init("empty_file_name")};
};
}

namespace {
struct LinalgCodegenPass : public PassWrapper<LinalgCodegenPass, OperationPass<FuncOp>> {
  LinalgCodegenPass() = default;
  LinalgCodegenPass(Options &options) {
    params.targetCPU = options.targetCPU;
    params.vectorWidth = options.vectorWidth;
    params.compileOptions = options.compileOptions;
  }
  LinalgCodegenPass(const LinalgCodegenPass &pass) {
    params = pass.params;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    func::FuncDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  template <typename LinalgNamedOp>
  void applyStrategyToNamedLinalgOp();

  void runOnOperation() override;

  struct Parameters {
    std::string vectorWidth, targetCPU;
    std::string compileOptions;
    bool fuse, fusePadding, licm, hoistRedundantVectorTransfers, vectorizePadding;
    bool vectorTransferPartialRewrite, vectorContractLowering, vectorToSCFConversion;
    int hoistPadding;
  };
  Parameters params;

  void runStrategy(Nod::OptionsT& options,
                   Parameters& params,
                   StringRef anchorOpName);

  Nod::CompileOptionsT config;
};
}  // namespace

void performTileOptions(Nod::OptionsT& options,
                        LinalgTilingOptions& tilingOptions,
                        LinalgTilingAndFusionOptions tilingAndFusionOptions,
                        bool& fuse,
                        bool& promote,
                        bool& promote_full_tile)
{
  const auto& tileOptions = options.tile_options;
  // Tile Codegen Options
  if (tileOptions != NULL) {
    linalg::LinalgTilingLoopType loop_type = linalg::LinalgTilingLoopType::Loops;
    switch(tileOptions->loop_type) {
      case Nod::LinalgTilingLoopType_loops:
        loop_type = linalg::LinalgTilingLoopType::Loops;
        break;
        case Nod::LinalgTilingLoopType_affine_loops:
        loop_type = linalg::LinalgTilingLoopType::AffineLoops;
        break;
        case Nod::LinalgTilingLoopType_parallel_loops:
        loop_type = linalg::LinalgTilingLoopType::ParallelLoops;
        break;
    }

    llvm::SmallVector<int64_t, 4> tileSizes;
    llvm::SmallVector<int64_t, 4> tileInterchange;

    for (int i = 0; i < tileOptions->tile_sizes.size(); i++) {
      tileSizes.push_back(tileOptions->tile_sizes[i]);
    }
    for (int i = 0; i < tileOptions->tile_interchange.size(); i++) {
      tileInterchange.push_back(tileOptions->tile_interchange[i]);
    }

    fuse = tileOptions->fuse;
    if (fuse){
      if (!tileSizes.empty()){
        tilingAndFusionOptions.tileSizes = {tileSizes.begin(), tileSizes.end()};
      }
      if (!tileInterchange.empty()){
        tilingAndFusionOptions.tileInterchange = {tileInterchange.begin(), tileInterchange.end()};
      }
    }
    else{
      if (!tileSizes.empty()){
        tilingOptions = tilingOptions.setLoopType(loop_type);
        tilingOptions = tilingOptions.setTileSizes(tileSizes);
      }
      if (!tileInterchange.empty()){
        tilingOptions = tilingOptions.setInterchange(
                            SmallVector<unsigned>(tileInterchange.begin(), tileInterchange.end()));
      }
    }
    promote = tileOptions->promote;
    promote_full_tile = tileOptions->promote_full_tile;
  }
}

void performPaddingOptions(Nod::OptionsT& options,
                           LinalgPaddingOptions& paddingOptions,
                           bool& pad)
{
  const auto& padOptions = options.pad_options;
  // Padding Codegen Options
  if (padOptions != NULL) {
    pad = padOptions->pad;
    llvm::SmallVector<int64_t, 4> pack_paddings;
    llvm::SmallVector<int64_t, 4> hoist_paddings;

    for (int i = 0; i < padOptions->pack_paddings.size(); i++) {
      pack_paddings.push_back(padOptions->pack_paddings[i]);
    }
    for (int i = 0; i < padOptions->hoist_paddings.size(); i++) {
      hoist_paddings.push_back(padOptions->hoist_paddings[i]);
    }


    auto packFunc = [&](OpOperand &opOperand) {
      return opOperand.getOperandNumber() < pack_paddings.size()
             ? pack_paddings[opOperand.getOperandNumber()]
             : false;
    };
    paddingOptions = paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
    auto hoistingFunc = [&](OpOperand &opOperand) {
      return opOperand.getOperandNumber() < hoist_paddings.size()
             ? hoist_paddings[opOperand.getOperandNumber()]
             : 0;
    };
    paddingOptions = paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);
  }
}

void performVectorizeOptions(Nod::OptionsT& options,
                             vector::VectorContractLowering& vectorContractLowering,
                             vector::VectorTransferSplit& vectorTransferSplit,
                             bool& unrollVectorTransfers)
{
  const auto& vectorizeOptions = options.vectorize_options;
  // Vectorize Codegen Options
  if (vectorizeOptions != NULL) {
    switch(vectorizeOptions->vectorize_to) {
      case Nod::VectorContractLowering_dot:
        vectorContractLowering = vector::VectorContractLowering::Dot;
        break;
      case Nod::VectorContractLowering_matmul:
        vectorContractLowering = vector::VectorContractLowering::Matmul;
        break;
      case Nod::VectorContractLowering_outer_product:
        vectorContractLowering = vector::VectorContractLowering::OuterProduct;
        break;
      default:
        vectorContractLowering = vector::VectorContractLowering::Dot;
        break;
    }

    switch(vectorizeOptions->vector_transfer_split) {
      case Nod::VectorTransferSplit_none:
        vectorTransferSplit = vector::VectorTransferSplit::None;
        break;
      case Nod::VectorTransferSplit_linalg_copy:
        vectorTransferSplit = vector::VectorTransferSplit::LinalgCopy;
        break;
      case Nod::VectorTransferSplit_vector_transfer:
        vectorTransferSplit = vector::VectorTransferSplit::VectorTransfer;
        break;
      default:
        vectorTransferSplit = vector::VectorTransferSplit::None;
        break;
    }
    unrollVectorTransfers = vectorizeOptions->unroll_vector_transfers;
  }
}

void LinalgCodegenPass::runStrategy(Nod::OptionsT& options,
                                    Parameters& params,
                                    StringRef anchorOpName) {
  CodegenStrategy strategy;
  LinalgTilingOptions tilingOptions;
  LinalgTilingAndFusionOptions tilingAndFusionOptions;
  LinalgPaddingOptions paddingOptions;
  vector::VectorContractLowering vectorContractLowering;
  vector::VectorTransferSplit vectorTransferSplit;
  bool unrollVectorTransfers, promote, promote_full_tile, pad, fuse;

  performTileOptions(options, tilingOptions, tilingAndFusionOptions, fuse, promote, promote_full_tile);
  //performPaddingOptions(options, paddingOptions, pad);
  performVectorizeOptions(options, vectorContractLowering, vectorTransferSplit, unrollVectorTransfers);

  strategy.tileIf(!fuse && options.tile_options != NULL, anchorOpName, tilingOptions)
          .tileAndFuseIf(fuse && options.tile_options != NULL, anchorOpName, tilingAndFusionOptions)
          .promoteIf(promote, anchorOpName,
                     LinalgPromotionOptions()
                        .setAlignment(16)
                        .setUseFullTileBuffersByDefault(promote_full_tile))
          //.padIf(pad, anchorOpName, paddingOptions)
          .vectorizeIf(options.vectorize_options != NULL, anchorOpName)
          .vectorLowering(
            LinalgVectorLoweringOptions()
              .setVectorTransformsOptions(
                  vector::VectorTransformsOptions()
                      .setVectorTransformsOptions(vectorContractLowering)
                      .setVectorTransferSplit(vectorTransferSplit))
              .setVectorTransferToSCFOptions(
                  VectorTransferToSCFOptions().enableFullUnroll(
                      unrollVectorTransfers))
              .enableTransferPartialRewrite()
              .enableContractionLowering()
              .enableTransferToSCFConversion());

  // Created a nested OpPassManager and run.
  FuncOp funcOp = getOperation();
  OpPassManager dynamicPM("builtin.func");
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}

void LinalgCodegenPass::runOnOperation() {
  MLIRContext *ctx = getOperation().getContext();
  SmallVector<Attribute, 4> attrs;
  attrs.push_back(ArrayAttr::get(ctx,
                                 {StringAttr::get(ctx, "prefer-vector-width"),
                                  StringAttr::get(ctx, params.vectorWidth)}
                                ));
  attrs.push_back(ArrayAttr::get(ctx,
                                 {StringAttr::get(ctx, "target-cpu"),
                                  StringAttr::get(ctx, params.targetCPU)}
                                ));
  getOperation()->setAttr("passthrough", ArrayAttr::get(ctx, attrs));

  // Load CompileOptions from file
  std::ifstream infile;
  infile.open(params.compileOptions, std::ios::binary | std::ios::in);
  infile.seekg(0,std::ios::end);
  int length = infile.tellg();
  if (length < 0) {
    std::cout << "File not found!" << std::endl;
    std::cout << params.compileOptions << std::endl;
    exit(1);
  }
  infile.seekg(0,std::ios::beg);
  char *data = new char[length];
  infile.read(data, length);
  infile.close();
  Nod::GetCompileOptions(data)->UnPackTo(&config);

  // Dynamic Codegen
  const auto& options = config.options;
  for (unsigned int option_index = 0; option_index < options.size(); option_index++) {
    const auto& option = options[option_index];
    StringRef anchorOpName;

    // TODO: add matmul column major op
    switch (option->op) {
      case Nod::LinalgOperator_matmul:
        anchorOpName = "linalg.matmul";
        runStrategy(*option, params, anchorOpName);
        break;
      case Nod::LinalgOperator_fill:
        anchorOpName = "linalg.fill";
        runStrategy(*option, params, anchorOpName);
        break;
      case Nod::LinalgOperator_copy:
        anchorOpName = "linalg.copy";
        runStrategy(*option, params, anchorOpName);
        break;
      case Nod::LinalgOperator_unknown:
        std::cout << "Must define operator in compile options" << std::endl;
        exit(1);
    }
  }
}

std::unique_ptr<OperationPass<FuncOp>> createLinalgCodegenPass(Options &options) {
  return std::make_unique<LinalgCodegenPass>(options);
}
}

//===----------------------------------------------------------------------===//

/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<StringError>(message.str(), llvm::inconvertibleErrorCode());
}

Error compile(Options &options, mlir::DialectRegistry &registry) {
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  llvm::errs() << "Read file: " << options.inputFile << "\n";
  OwningOpRef<mlir::ModuleOp> moduleRef = parseSourceFile<mlir::ModuleOp>(options.inputFile, &context);
  if (!moduleRef)
    return make_string_error(Twine("could not open ") + options.inputFile);

  ModuleOp module = *moduleRef;
  PassManager pm(module.getContext(), OpPassManager::Nesting::Implicit);
  mlir::applyPassManagerCLOptions(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLinalgCodegenPass(options));
  pm.addPass(createLinalgComprehensiveModuleBufferizePass());

  // Lower to LLVM
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertLinalgToLLVMPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createMemRefToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (failed(pm.run(module))) {
    return make_string_error(Twine("error compiling to llvm backend"));
  }

  // Convert from MLIR to LLVMIR
  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return make_string_error(Twine("error translating to llvm ir"));
  }

  std::string moduleStr;
  llvm::raw_string_ostream ss(moduleStr);
  ss << *llvmModule;

  std::ofstream output(std::string(options.inputFile) + ".ll");
  output << moduleStr;
  output.close();

  return Error::success();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllPasses();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  mlir::initializeLLVMPasses();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();

  Options options;
  llvm::cl::ParseCommandLineOptions(argc, argv, "matmul-compile\n");

  auto error = compile(options, registry);
  int exitCode = EXIT_SUCCESS;
  llvm::handleAllErrors(std::move(error), [&exitCode](const llvm::ErrorInfoBase &info) {
    llvm::errs() << "Error: ";
    info.log(llvm::errs());
    llvm::errs() << '\n';
    exitCode = EXIT_FAILURE;
  });
  return exitCode;
}
