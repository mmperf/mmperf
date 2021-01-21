#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace mlir;
using namespace mlir::linalg;
using llvm::Error;
using llvm::Expected;
using llvm::StringError;
using llvm::Twine;

//===----------------------------------------------------------------------===//
// Linalg codegen pass
// This follows the Custom BLIS schedule on slide 133 of [1]
//
//
// [1] https://drive.google.com/file/d/1_zPPxOILAIHOWoSM7GALwioYOGEgD2Xe/view?usp=sharing
namespace mlir {

namespace {
namespace cl = llvm::cl;
struct Options {
  cl::opt<std::string> inputFile{cl::Positional,
    cl::desc("the input .mlir file"),
    cl::init("")};

  // Matrix multiplication sizes
  cl::opt<int> M{"M", cl::Required,
    cl::desc("Number of rows of first matrix"),
    cl::init(1024)};
  cl::opt<int> N{"N", cl::Required,
    cl::desc("Number of rows of first matrix"),
    cl::init(1024)};
  cl::opt<int> K{"K", cl::Required,
    cl::desc("Number of rows of first matrix"),
    cl::init(1024)};

  // CPU info
  cl::opt<std::string> targetCPU{"target-cpu", cl::Required,
    cl::desc("Target CPU for codegen"),
    cl::init("skylake-avx512")};
  cl::opt<std::string> vectorWidth{"vector-width", cl::Required,
    cl::desc("Target vector width for codegen"),
    cl::init("512")};

  // Codegen info
  cl::opt<std::string> registerTileSizes{"register-tile-sizes",
    cl::desc("'x'-separated triple. Specifies the size of the register "
             "tile that will be use to vectorize"),
    cl::init("")};

  cl::opt<bool> promote{"promote",
    cl::desc("Promote the registerTile into a small aligned scratchpad region."),
    cl::init(false)};

  cl::opt<bool> promoteFullTile{"promote-full-tile-pad",
    cl::desc("Pad the small aligned scratchpad region to the "
             "registerTiling size. This enables explicit vectorization "
             "even in symbolic cases (but has a cost)."),
    cl::init(true)};

  cl::opt<bool> vectorize{"vectorize",
    cl::desc("Rewrite the registerTile as a vector operation."),
    cl::init(false)};

  cl::opt<std::string> vectorizeTo{"vectorize-to",
    cl::desc("the type of vector op to use"),
    cl::init("outerproduct")};

  cl::opt<std::string> splitVectorTransfersTo{"split-vector-transfers-to",
    cl::desc("The kind of of op to split vector transfers to"),
    cl::init("vector-transfers")};

  cl::opt<bool> unrollVectorTransfers{"unroll-vector-transfers",
    cl::desc("Enable full unrolling of vector.transfer operations"),
    cl::init(false)};
};
}

namespace {
struct LinalgCodegenPass : public PassWrapper<LinalgCodegenPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AffineDialect, scf::SCFDialect>();
  }
  LinalgCodegenPass() = default;
  LinalgCodegenPass(Options &options) {
    params.M = options.M;
    params.N = options.N;
    params.K = options.K;
    params.targetCPU = options.targetCPU;
    params.vectorWidth = options.vectorWidth;
    params.registerTileSizes = options.registerTileSizes;
    params.promote = options.promote;
    params.promoteFullTile = options.promoteFullTile;
    params.vectorize = options.vectorize;
    params.vectorizeTo = options.vectorizeTo;
    params.splitVectorTransfersTo = options.splitVectorTransfersTo;
    params.unrollVectorTransfers = options.unrollVectorTransfers;
  }
  LinalgCodegenPass(const LinalgCodegenPass &pass) {
    params = pass.params;
  }
  void runOnFunction() override;

  struct Parameters {
    int M, N, K;
    std::string vectorWidth, targetCPU;
    std::string registerTileSizes;
    bool promote;
    bool promoteFullTile;
    bool vectorize;
    std::string vectorizeTo;
    std::string splitVectorTransfersTo;
    bool unrollVectorTransfers;
  };
  Parameters params;
};
}  // namespace

void LinalgCodegenPass::runOnFunction() {
  MLIRContext *ctx = getFunction().getContext();
  SmallVector<Attribute, 4> attrs;
  attrs.push_back(ArrayAttr::get({StringAttr::get("prefer-vector-width", ctx),
                                  StringAttr::get(params.vectorWidth, ctx)},
                                  ctx));
  attrs.push_back(ArrayAttr::get({StringAttr::get("target-cpu", ctx),
                                  StringAttr::get(params.targetCPU, ctx)},
                                  ctx));
  getFunction()->setAttr("passthrough", ArrayAttr::get(attrs, ctx));

  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          params.vectorizeTo)
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          params.splitVectorTransfersTo)
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  // Small and medium codegen
  if (params.M < 1000) {
    LinalgTilingOptions tilingOptions;
    llvm::SmallVector<int64_t, 4> tileSizes{6, 16, 16};
    if (!tileSizes.empty())
      tilingOptions = tilingOptions.setTileSizes(tileSizes);

    CodegenStrategy strategy;
    strategy.tile<MatmulOp>(tilingOptions)
        .promote<MatmulOp>(LinalgPromotionOptions()
			      .setAlignment(16)
			      .setUseFullTileBuffersByDefault(true))
        .vectorize<MatmulOp>()
        .setVectorTransformsOptions(
            vector::VectorTransformsOptions()
                .setVectorTransformsOptions(vectorContractLowering)
                .setVectorTransferSplit(vectorTransferSplit))
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(params.unrollVectorTransfers));

    strategy.transform(getFunction());
  }

  // Large codegen
  if (params.M > 1000) {
    // Step 1: tile, interchange and promote A and B. Copy of A gets hoisted above j.
    // TODO: Could not find option to outline matmul op to lower register pressure?
    {
      CodegenStrategy strategyCaches;
      strategyCaches
        .tile<MatmulOp>(LinalgTilingOptions()
			.setTileSizes({128, 128, 256})
			.setInterchange({0, 2, 1}))
        .promote<MatmulOp>(LinalgPromotionOptions()
			   .setOperandsToPromote({0, 1})
			   .setAlignment(getpagesize()));

      strategyCaches.transform(getFunction());
    }

    // Step 2: a simple 2-D copy vectorization on the non-outlined copies.
    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<FillOp>(LinalgTilingOptions().setTileSizes({4, 16}))
        .vectorize<FillOp>()
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(params.unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }

    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<CopyOp>(LinalgTilingOptions().setTileSizes({4, 16}))
        .vectorize<CopyOp>()
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(params.unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }

    // Step 3: apply the register level strategy on the outlined piece.
    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<MatmulOp>(LinalgTilingOptions().setTileSizes({8, 16, 8}))
        .promote<MatmulOp>(LinalgPromotionOptions()
                             .setUseFullTileBuffersByDefault(params.promoteFullTile)
                             .setAlignment(128))
        .vectorize<MatmulOp>()
        .setVectorTransformsOptions(
            vector::VectorTransformsOptions()
                .setVectorTransformsOptions(vectorContractLowering)
                .setVectorTransferSplit(vectorTransferSplit))
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(params.unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
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
  MLIRContext context;
  registry.loadAll(&context);
  llvm::errs() << "Read file: " << options.inputFile << "\n";
  OwningModuleRef moduleRef = parseSourceFile(options.inputFile, &context);
  if (!moduleRef)
    return make_string_error(Twine("could not open ") + options.inputFile);

  ModuleOp module = *moduleRef;
  PassManager pm(module.getContext(), OpPassManager::Nesting::Implicit);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createLinalgCodegenPass(options));

  // Lower to LLVM
  pm.addPass(createConvertVectorToSCFPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerToCFGPass());
  pm.addPass(createConvertVectorToLLVMPass());
  pm.addPass(createLowerToLLVMPass());

  if (failed(pm.run(module))) {
    return make_string_error(Twine("error compiling to llvm backend"));
  }

  // Convert from MLIR to LLVMIR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return make_string_error(Twine("error translating to llvm ir"));
  }

  std::string moduleStr;
  llvm::raw_string_ostream ss(moduleStr);
  ss << *llvmModule;

  std::string name = std::filesystem::path(std::string(options.inputFile)).stem();
  name += ".ll";
  std::ofstream output(name);
  output << moduleStr;
  output.close();

  return Error::success();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
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
