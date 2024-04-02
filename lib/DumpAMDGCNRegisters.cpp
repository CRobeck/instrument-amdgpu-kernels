#include "DumpAMDGCNRegisters.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
using namespace llvm;

bool DumpAMDGCNRegisters::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  InlineAsm* V;
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
        // for(Function::iterator BB = F.begin();BB!=F.end();BB++){
        //   for(BasicBlock::iterator I = BB->begin();I!=BB->end();I++){  
        //     if (isa<CallInst>(I) || isa<InvokeInst>(I)) {
        //       const CallBase &cb = cast<CallBase>(*I);
        //       if (const InlineAsm *IA = dyn_cast<InlineAsm>(cb.getCalledOperand())) {
        //         const std::string &AsmString = IA->getAsmString();
        //         const std::string &ConstraintStr = IA->getConstraintString();
        //         errs() <<  *IA << "\n";  
        //         errs() << "AsmString: " << AsmString << "\n";  
        //         errs() << "ConstraintStr: " << ConstraintStr << "\n"; 
        //         errs() << *IA->getFunctionType() << "\n";   
                // for (llvm::Function::arg_iterator arg = kernelFunc->arg_begin();
                //       arg != kernelFunc->arg_end(); ++arg)                
                // V = InlineAsm::get(IA->getFunctionType(), IA->getAsmString(), IA->getConstraintString(),
                //                    IA->hasSideEffects(), IA->isAlignStack(),
                //                    IA->getDialect(), IA->canThrow()); 
            }
          }
        }
    }             

      FunctionType *FTy = FunctionType::get(Type::getInt32Ty(CTX), Type::getInt32Ty(CTX), false);
      // std::string AsmString = "v_mov_b32 $0 v0\n";
      std::string AsmString = "s_mov_b32 m0 $1\ns_mov_b32 $0 m0\n";

      InlineAsm* InlineAsmFunc = InlineAsm::get(FTy, AsmString, "=s,I", false); 
           
      errs() << *FTy << "\n";   
      // // Get an IR builder. Sets the insertion point to the top of the function
      IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());
      Value* UniqueInt = Builder.getInt32(5);

      Value *InlineAsmFuncResult = Builder.CreateCall(InlineAsmFunc, {UniqueInt});
      Builder.CreateCall(InjectedFunction, {InlineAsmFuncResult});

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
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(DumpAMDGCNRegisters());
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "inject-amdgcn-func", LLVM_VERSION_STRING,
          callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
