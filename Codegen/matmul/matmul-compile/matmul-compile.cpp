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

#define STRING(s) #s
#define TO_STRING(x) STRING(x)

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
struct LinalgCodegenPass : public PassWrapper<LinalgCodegenPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, AffineDialect, scf::SCFDialect>();
  }
  LinalgCodegenPass() = default;
  LinalgCodegenPass(int M, int N, int K, const std::string &target_cpu,
		    const std::string &vector_width) : M(M), N(N), K(K),
		    target_cpu(target_cpu), vector_width(vector_width) {}
  LinalgCodegenPass(const LinalgCodegenPass &pass) {
    M = pass.M;
    N = pass.N;
    K = pass.K;
    target_cpu = pass.target_cpu;
    vector_width = pass.vector_width;
  }
  void runOnFunction() override;

  int M, N, K;
  std::string target_cpu, vector_width;
};
}  // namespace

void LinalgCodegenPass::runOnFunction() {
  MLIRContext *ctx = getFunction().getContext();
  SmallVector<Attribute, 4> attrs;
  attrs.push_back(ArrayAttr::get({StringAttr::get("prefer-vector-width", ctx),
                                  StringAttr::get(vector_width, ctx)},
                                  ctx));
  attrs.push_back(ArrayAttr::get({StringAttr::get("target-cpu", ctx),
                                  StringAttr::get(target_cpu, ctx)},
                                  ctx));
  getFunction()->setAttr("passthrough", ArrayAttr::get(attrs, ctx));

  std::string vectorizeContractionTo("outerproduct");
  std::string splitVectorTransfersTo("vector-transfers");
  bool registerPromoteFullTile{true};
  bool unrollVectorTransfers{true};
  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          vectorizeContractionTo)
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo)
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  // Small and medium codegen
  if (M < 1000) {
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
            VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));

    strategy.transform(getFunction());
  }

  // Large codegen
  if (M > 1000) {
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
            VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }

    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<CopyOp>(LinalgTilingOptions().setTileSizes({4, 16}))
        .vectorize<CopyOp>()
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }

    // Step 3: apply the register level strategy on the outlined piece.
    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<MatmulOp>(LinalgTilingOptions().setTileSizes({8, 16, 8}))
        .promote<MatmulOp>(LinalgPromotionOptions()
                             .setUseFullTileBuffersByDefault(registerPromoteFullTile)
			   .setAlignment(128))
        .vectorize<MatmulOp>()
        .setVectorTransformsOptions(
            vector::VectorTransformsOptions()
                .setVectorTransformsOptions(vectorContractLowering)
                .setVectorTransferSplit(vectorTransferSplit))
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }
  }

}

std::unique_ptr<OperationPass<FuncOp>> createLinalgCodegenPass(int M, int N, int K,
		const std::string &target_cpu, const std::string &vector_width) {
  return std::make_unique<LinalgCodegenPass>(M, N, K, target_cpu, vector_width);
}

}

//===----------------------------------------------------------------------===//


/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<StringError>(message.str(), llvm::inconvertibleErrorCode());
}

namespace {
namespace cl = llvm::cl;
struct Options {
  cl::opt<std::string> inputFile{cl::Positional, cl::desc("the input .mlir file"), cl::init("-")};
};
}

static void get_dimensions(const std::string filename, int &M, int &N, int &K) {
  std::stringstream ss(filename.substr(filename.find_last_of("/")));
  std::string token;
  std::vector<std::string> tokens;
  // Name is matmul_1x2x3.mlir
  while (std::getline(ss, token, '_')) {
    tokens.push_back(token);
  }

  std::string lastToken = tokens.back();
  std::string sizes = lastToken.substr(0, lastToken.find_last_of("."));
  std::stringstream nss(sizes);
  std::vector<int> sizeVec;
  while (std::getline(nss, token, 'x')) {
    sizeVec.push_back(std::stoi(token));
  }

  M = sizeVec[0];
  N = sizeVec[1];
  K = sizeVec[2];
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
  int M, N, K;
  get_dimensions(options.inputFile, M, N, K);
  pm.addPass(createCanonicalizerPass());
  std::string target_cpu = TO_STRING(TARGET_CPU);
  std::string vector_width = TO_STRING(VECTOR_WIDTH);
  pm.addPass(createLinalgCodegenPass(M, N, K, target_cpu, vector_width));

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
