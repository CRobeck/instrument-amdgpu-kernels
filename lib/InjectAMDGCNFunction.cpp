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
#if LLVM_VERSION_MAJOR >= 20
      Function *WorkItemXIDIntrinsicFunc = cast<Function>(
          M.getOrInsertFunction(
               Intrinsic::getName(Intrinsic::amdgcn_workitem_id_x,
                                  {Type::getInt32Ty(CTX)}, &M),
               FunctionType::get(Type::getInt32Ty(CTX), false))
              .getCallee());

#elif LLVM_VERSION_MAJOR == 19
      Function *WorkItemXIDIntrinsicFunc = Intrinsic::getOrInsertDeclaration(
          F.getParent(), Intrinsic::amdgcn_workitem_id_x);
#else
      Function *WorkItemXIDIntrinsicFunc = Intrinsic::getDeclaration(
          F.getParent(), Intrinsic::amdgcn_workitem_id_x);
#endif

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
        [&](ModulePassManager &MPM, auto &&...args) {
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
