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
  IRBuilder<> ModuleBuilder(CTX);
  Value* TtraceCounter = ModuleBuilder.getInt32(0);
  unsigned UniqueInt = 0;
//  if(!InjectedTtraceCounter)
//	errs() << "Global var not found" << "\n";   
//  GlobalVariable *TtraceCounter = 
//      new GlobalVariable(/*Module=*/M,
//                         /*Type=*/Type::getInt32Ty(CTX),
//                         /*isConstant=*/false,
//                         /*Linkage=*/GlobalValue::InternalLinkage,
//                         /*Initializer=*/0, 
//                         /*Name=*/"ttrace_counter");
  // InlineAsm* V;
  for (auto &F : M) {
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
        for(Function::iterator BB = F.begin();BB!=F.end();BB++){
          for(BasicBlock::iterator I = BB->begin();I!=BB->end();I++){  
			// if (auto LI = dyn_cast<LoadInst>(I)){
      //   GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(LI->getPointerOperand());
      //   Value *Op = GEPInst->getPointerOperand()->stripPointerCasts();
			// 	unsigned AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
      //         	errs() << *LI << "\n";			
			// 	errs() << "Address Space: " << AddrSpace << "\n";
      //   if (AddrSpace == 3){
      //       FunctionType *FTy = FunctionType::get(Type::getVoidTy(CTX), false);
      //       std::string AsmString = "s_mov_b32 m0 " + std::to_string(UniqueInt) + "\n";
      //       IRBuilder<> Builder(LI);
      //       InlineAsm* InlineAsmFunc = InlineAsm::get(FTy, AsmString, "", false);
      //       Builder.CreateCall(InlineAsmFunc, {});
      //       I++;
      //       Builder.SetInsertPoint(dyn_cast<Instruction>(I));
      //       I--;
      //       Builder.CreateCall(InlineAsm::get(FTy, "s_mov_b32 m0 53\n", "", false), {});            
      //       UniqueInt++;
      //   }
      //       }

			if (auto SI = dyn_cast<StoreInst>(I)){
        GetElementPtrInst *GEPInst = dyn_cast<GetElementPtrInst>(SI->getPointerOperand());
        Value *Op = GEPInst->getPointerOperand()->stripPointerCasts();
				unsigned AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
              	errs() << *SI << "\n";			
				errs() << "Address Space: " << AddrSpace << "\n";
        if (AddrSpace == 3){
            FunctionType *FTy = FunctionType::get(Type::getInt32Ty(CTX), true);
//			GlobalVariable *TtraceCounter = M.getNamedGlobal("ttrace_counter");
            // std::string AsmString = "s_mov_b32 m0 " + std::to_string(UniqueInt) + "\n""s_nop 0\n""s_mov_b32 m0 53\n";
            std::string AsmString = "s_mov_b32 $0 m0\ns_nop 0\n""s_mov_b32 m0 $1""\n""s_nop 0\n""s_ttracedata\n""s_nop 0\n""s_mov_b32 m0 $0\n""s_nop 0\n""s_add_i32 $1 $1 1\n";
            IRBuilder<> Builder(SI);
            InlineAsm* InlineAsmFunc = InlineAsm::get(FTy, AsmString, "=s,s", true);
            Builder.CreateCall(InlineAsmFunc, {TtraceCounter});
            UniqueInt++;
            return true;
        }
            }


        }
    } //End of instructions in AMDGCN kernel loop

    errs() << "Injected " << UniqueInt << " LDS Load/Store s_ttrace instructions\n"; 
       

      // FunctionType *FTy = FunctionType::get(Type::getInt32Ty(CTX), Type::getInt32Ty(CTX), false);
      // std::string AsmString = "s_mov_b32 m0 $1\ns_mov_b32 $0 m0\n";

      // InlineAsm* InlineAsmFunc = InlineAsm::get(FTy, AsmString, "=s,I", true); 
           
      // // // Get an IR builder. Sets the insertion point to the top of the function
      // IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());
      // Value* UniqueInt = Builder.getInt32(5);

      // Builder.CreateCall(InlineAsmFunc, {UniqueInt});
      // // Builder.CreateCall(InjectedFunction, {InlineAsmFuncResult});

      // errs() << "Injecting Device Function Into AMDGPU Kernel: " << F.getName()
      //        << "\n";

      ModifiedCodeGen = true;
    } //End of if AMDGCN Kernel
  } //End of functions in module loop
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
