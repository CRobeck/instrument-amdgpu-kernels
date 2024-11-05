#include "InjectAMDGCNFunction.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
using namespace llvm;

bool InjectAMDGCNFunc::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  for (auto &F : M) {
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      FunctionType *FT = FunctionType::get(Type::getVoidTy(CTX),
                                           {Type::getInt32Ty(CTX)}, false);
      FunctionCallee InjectedFunctionCallee =
          M.getOrInsertFunction("_Z11PrintKerneli", FT);
      Function *InjectedFunction =
          cast<Function>(InjectedFunctionCallee.getCallee());
      errs() << "Function To Be Injected: " << InjectedFunction->getName()
             << "\n";

      // Get an IR builder. Sets the insertion point to the top of the function
      IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());
      Function *WorkItemXIDIntrinsicFunc = Intrinsic::getOrInsertDeclaration(
          F.getParent(), Intrinsic::amdgcn_workitem_id_x);

      Value *WorkItemXValue = Builder.CreateCall(WorkItemXIDIntrinsicFunc, {});
      Builder.CreateCall(InjectedFunction, {WorkItemXValue});

      errs() << "Injecting Device Function Into AMDGPU Kernel: " << F.getName()
             << "\n";

      ModifiedCodeGen = true;
    }
  }
  return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, OptimizationLevel OL, ThinOrFullLTOPhase Phase) {
          MPM.addPass(InjectAMDGCNFunc());
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "inject-amdgcn-func", LLVM_VERSION_STRING,
          callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
