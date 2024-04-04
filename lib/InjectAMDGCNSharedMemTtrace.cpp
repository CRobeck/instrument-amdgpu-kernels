#include "InjectAMDGCNSharedMemTtrace.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
using namespace llvm;

bool InjectAMDGCNSharedMemTtrace::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  IRBuilder<> ModuleBuilder(CTX);
  Value* TtraceCounter = ModuleBuilder.getInt32(0);
  unsigned UniqueInt = 0;
  for (auto &F : M) {
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
        for(Function::iterator BB = F.begin();BB!=F.end();BB++){
          for(BasicBlock::iterator I = BB->begin();I!=BB->end();I++){  
			if (auto LI = dyn_cast<LoadInst>(I)){
        GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
        Value *Op = GEPInst->getPointerOperand()->stripPointerCasts();
				unsigned AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
        if (AddrSpace == 3){
            FunctionType *FTy = FunctionType::get(Type::getInt32Ty(CTX), true);
            std::string AsmString = "s_mov_b32 $0 m0\n""s_mov_b32 m0 $1""\n""s_nop 0\n""s_ttracedata\n""s_mov_b32 m0 $0\n""s_add_i32 $1 $1 1\n";
            IRBuilder<> Builder(LI);
            InlineAsm* InlineAsmFunc = InlineAsm::get(FTy, AsmString, "=s,s", true);
            Builder.CreateCall(InlineAsmFunc, {TtraceCounter});
            UniqueInt++;
            I++;
            Builder.SetInsertPoint(dyn_cast<Instruction>(I));
            Builder.CreateCall(InlineAsm::get(FTy, "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"\
                                                   "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"\
                                                   "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n", "", false), {});      
        }
            }

			if (auto SI = dyn_cast<StoreInst>(I)){
        GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
        Value *Op = GEPInst->getPointerOperand()->stripPointerCasts();
				unsigned AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
        if (AddrSpace == 3){
            FunctionType *FTy = FunctionType::get(Type::getInt32Ty(CTX), true);
            std::string AsmString = "s_mov_b32 $0 m0\n""s_mov_b32 m0 $1""\n""s_nop 0\n""s_ttracedata\n""s_mov_b32 m0 $0\n""s_add_i32 $1 $1 1\n";
            IRBuilder<> Builder(SI);
            InlineAsm* InlineAsmFunc = InlineAsm::get(FTy, AsmString, "=s,s", true);
            Builder.CreateCall(InlineAsmFunc, {TtraceCounter});
            UniqueInt++;
            I++;
            Builder.SetInsertPoint(dyn_cast<Instruction>(I));
            Builder.CreateCall(InlineAsm::get(FTy, "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"\
                                                   "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"\
                                                   "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n", "", false), {});      
        }
            }


        }
    } //End of instructions in AMDGCN kernel loop

    errs() << "Injected LDS Load/Store s_ttrace instructions at " << UniqueInt <<
               " source locations\n"; 
       

      ModifiedCodeGen = true;
    } //End of if AMDGCN Kernel
  } //End of functions in module loop
  return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerPipelineEarlySimplificationEPCallback(
        [&](ModulePassManager &MPM, auto) {
          MPM.addPass(InjectAMDGCNSharedMemTtrace());
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "inject-amdgcn-lds-ttrace", LLVM_VERSION_STRING,
          callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
