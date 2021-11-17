// Copyright 2021 Nod Labs
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <iostream>

#include "flatbuffers/flatbuffers.h"
#include "compile_options_generated.h"

using namespace mlir;

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

  // Codegen info
  cl::opt<std::string> compileOptions{"compile-options", cl::Required,
    cl::desc("Flatbuffer file describing compile options configurations"),
    cl::init("empty_file_name")};
};
}

namespace {
template<typename T>
void populateSmallVector(std::vector<T> vec, SmallVector<T> &smallvec) {
  for (auto el : vec) {
    smallvec.push_back(el);
  }
}

// Adds lowering.config attribute to mhlo.dot op for tiling
struct AddTiling : public OpRewritePattern<mhlo::DotOp> {
  using OpRewritePattern<mhlo::DotOp>::OpRewritePattern;

  AddTiling(MLIRContext *ctx,
            iree_compiler::TileSizesListType _tileSizes,
            SmallVector<int64_t> _nativeVectorSizes,
            SmallVector<int64_t> _workloadPerWorkgroup) : OpRewritePattern<mhlo::DotOp>(ctx),
            tileSizes(_tileSizes),
            nativeVectorSizes(_nativeVectorSizes),
            workloadPerWorkgroup(_workloadPerWorkgroup) {}

  LogicalResult matchAndRewrite(mhlo::DotOp op, PatternRewriter &rewriter) const override {
    // exit if op already has lowering.config attribute
    if (op->hasAttr("compilation.info")) {
      return failure();
    }

    auto compilationAttr = iree_compiler::IREE::Codegen::CompilationInfoAttr::get(
                          op->getContext(), tileSizes, nativeVectorSizes,
                          iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::CPUTileFuseAndVectorize,
                          workloadPerWorkgroup,
                          /*workgroupSize =*/ArrayRef<int64_t>{});

    // Currently, verification only works for pass pipeline 'CPUTensorToVectors'
    LogicalResult status = iree_compiler::verifyLoweringConfiguration(
            op, compilationAttr.getLoweringConfig(), compilationAttr.getTranslationInfo(),
            /*workgroupSize =*/ArrayRef<int64_t>{});
    if (failed(status)) return failure();

    rewriter.updateRootInPlace(op, [&]() {
        op->setAttr("compilation.info", compilationAttr);
    });

    return success();
  }
  iree_compiler::TileSizesListType tileSizes;
  SmallVector<int64_t> nativeVectorSizes;
  SmallVector<int64_t> workloadPerWorkgroup;
};

struct IREETilingPass : public PassWrapper<IREETilingPass, OperationPass<ModuleOp>> {
  IREETilingPass() = default;
  IREETilingPass(Options &options) {
    params.compileOptions = options.compileOptions;
  }
  IREETilingPass(const IREETilingPass &pass) {
    params = pass.params;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<iree_compiler::IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override {
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
    Nod::GetTileOptions(data)->UnPackTo(&config);

    // Set tiling vectors
    SmallVector<int64_t> workloadPerWorkgroup, L1TileSizes, nativeVectorSizes;
    populateSmallVector<int64_t>(config.work_group_tile_sizes, workloadPerWorkgroup);
    populateSmallVector<int64_t>(config.l1_tile_sizes, L1TileSizes);
    populateSmallVector<int64_t>(config.vector_tile_sizes, nativeVectorSizes);
    iree_compiler::TileSizesListType tileSizes = {{}, L1TileSizes, nativeVectorSizes};

    // Appy pattern rewrite to add lowering.config attribute to mhlo.dot op
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<AddTiling>(context, tileSizes, nativeVectorSizes, workloadPerWorkgroup);
    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  struct Parameters {
    std::string compileOptions;
  };
  Parameters params;
  Nod::TileOptionsT config;
};
}  // namespace

// pass registration
std::unique_ptr<OperationPass<ModuleOp>> createIREETilingPass(Options &options) {
    return std::make_unique<IREETilingPass>(options);
}
}

//===----------------------------------------------------------------------===//

/// Wrap a string into an llvm::StringError.
static Error make_string_error(const Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(), llvm::inconvertibleErrorCode());
}

Error compile(Options &options, mlir::DialectRegistry &registry) {
  MLIRContext context(registry);
  context.loadAllAvailableDialects();
  context.allowUnregisteredDialects();

  llvm::errs() << "Read file: " << options.inputFile << "\n";
  OwningModuleRef moduleRef = parseSourceFile(options.inputFile, &context);
  if (!moduleRef)
    return make_string_error(Twine("could not open ") + options.inputFile);

  ModuleOp module = *moduleRef;
  PassManager pm(&context);
  pm.addPass(createIREETilingPass(options));

  if (failed(pm.run(module))) {
    return make_string_error(Twine("error compiling to llvm backend"));
  }

  std::string moduleStr;
  llvm::raw_string_ostream ss(moduleStr);
  ss << *module;

  std::string name = std::filesystem::path(std::string(options.inputFile)).stem();
  name += "-tiled.mlir";
  std::ofstream output(name);
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
  llvm::cl::ParseCommandLineOptions(argc, argv, "matmul-iree-compile\n");

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
