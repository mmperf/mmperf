// Copyright 2021 Nod Labs
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Error.h"
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
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
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

#define kNumMaxParallelDims 3

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

SmallVector<int> getMatmulSize(Operation* op) {
    auto linalg_matmul = llvm::dyn_cast<linalg::MatmulOp>(op);
    auto mhlo_dot = llvm::dyn_cast<mhlo::DotOp>(op);
    ArrayRef<int64_t> lhs_shape;
    ArrayRef<int64_t> rhs_shape;
    ArrayRef<int64_t> out_shape;
    if(mhlo_dot) {
      mhlo::DotOp dot_op = llvm::dyn_cast<mhlo::DotOp>(op);
      lhs_shape = dot_op.lhs().getType().dyn_cast<RankedTensorType>().getShape();
      rhs_shape = dot_op.rhs().getType().dyn_cast<RankedTensorType>().getShape();
      out_shape = dot_op.getResult().getType().dyn_cast<RankedTensorType>().getShape();
    } else {
      linalg::MatmulOp linalg_op = llvm::dyn_cast<linalg::MatmulOp>(op);
      lhs_shape = linalg_op.inputs()[0].getType().dyn_cast<RankedTensorType>().getShape();
      rhs_shape = linalg_op.inputs()[1].getType().dyn_cast<RankedTensorType>().getShape();
      out_shape = linalg_op.result_tensors()[0].getType().dyn_cast<RankedTensorType>().getShape();
    }
    int m = out_shape[0];
    int n = out_shape[1];
    int k = lhs_shape[1];
    return {m, n, k};
}

struct IREETilingPass : public PassWrapper<IREETilingPass, OperationPass<ModuleOp>> {
  IREETilingPass() = default;
  IREETilingPass(Options &options) {
    params.compileOptions = options.compileOptions;
  }
  IREETilingPass(const IREETilingPass &pass) {
    params = pass.params;
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
    Nod::GetCompileOptions(data)->UnPackTo(&config);

    // Appy pattern rewrite to add lowering.config attribute to op
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    const auto& options = config.options;
    bool tagging = false;
    if(config.m && config.n && config.k) {
      tagging = true;
    }
    int count = 0;

    moduleOp->walk([&](Operation *nestedOp) {
      auto linalg_matmul = llvm::dyn_cast<linalg::MatmulOp>(nestedOp);
      auto mhlo_dot = llvm::dyn_cast<mhlo::DotOp>(nestedOp);

      if(mhlo_dot || linalg_matmul) {
        const auto& option = (count < options.size()) ? options[count++] : options.back();
        auto op = mhlo_dot ? mhlo_dot : linalg_matmul;

        // Set tiling vectors
        SmallVector<int64_t> workloadPerWorkgroup, L1TileSizes, vectorTileSizes, workgroupSizes;
        populateSmallVector<int64_t>(option->work_group_tile_sizes, workloadPerWorkgroup);
        populateSmallVector<int64_t>(option->l1_tile_sizes, L1TileSizes);
        populateSmallVector<int64_t>(option->vector_tile_sizes, vectorTileSizes);
        populateSmallVector<int64_t>(option->work_group_sizes, workgroupSizes);

        // set DispatchLoweringPassPipeline
        iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline passPipeline;
        iree_compiler::TileSizesListType tileSizes;
        switch(option->pipeline) {
          case Nod::PipelineType_CPU:
            passPipeline = iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
            tileSizes = {{}, L1TileSizes, vectorTileSizes};
            std::cout << "Using CPUDoubleTilingExpert pass pipeline" << std::endl;
            break;
          case Nod::PipelineType_GPU:
            passPipeline = iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt;
            tileSizes = {workloadPerWorkgroup};
            workloadPerWorkgroup.pop_back();
            std::reverse(workloadPerWorkgroup.begin(), workloadPerWorkgroup.end());
            std::cout << "Using LLVMGPUMatmulSimt pass pipeline" << std::endl;
            break;
          case Nod::PipelineType_GPU_TENSORCORE:
            passPipeline = iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulTensorCore;
            tileSizes = {workloadPerWorkgroup};
            workloadPerWorkgroup.pop_back();
            std::reverse(workloadPerWorkgroup.begin(), workloadPerWorkgroup.end());
            std::cout << "Using LLVMGPUTensorCore pass pipeline" << std::endl;
            break;
          default:
            passPipeline = iree_compiler::IREE::Codegen::DispatchLoweringPassPipeline::CPUDoubleTilingExpert;
            tileSizes = {{}, L1TileSizes, vectorTileSizes};
            std::cout << "Default using CPU pass pipeline" << std::endl;
            break;
        }

        auto compilationAttr = iree_compiler::IREE::Codegen::CompilationInfoAttr::get(
                               op->getContext(), tileSizes, /*nativeVectorSizes=*/ArrayRef<int64_t>{},
                               passPipeline, workloadPerWorkgroup, workgroupSizes);

        LogicalResult status = iree_compiler::verifyLoweringConfiguration(
                                  op, compilationAttr.getLoweringConfig(),
                                  compilationAttr.getTranslationInfo(), workgroupSizes);
        if (failed(status)) signalPassFailure();
        if(tagging) {
          SmallVector<int>  mm_shape = getMatmulSize(op);
          if(config.m == mm_shape[0] && config.n == mm_shape[1] && config.k == mm_shape[2]) {
            op->setAttr("compilation.info", compilationAttr);
          }
        } else {
          op->setAttr("compilation.info", compilationAttr);
        }
      }
    });
  }

  struct Parameters {
    std::string compileOptions;
  };
  Parameters params;
  Nod::CompileOptionsT config;
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
  OwningOpRef<mlir::ModuleOp> moduleRef = parseSourceFile(options.inputFile, &context);
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
  registry.insert<iree_compiler::IREE::Codegen::IREECodegenDialect,
                  iree_compiler::IREE::Util::UtilDialect,
                  iree_compiler::IREE::HAL::HALDialect,
                  mlir::mhlo::MhloDialect>();
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
