#include "AMDGCNNumCacheLines.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/Support/Allocator.h"
#include <iostream>
#include <vector>
using namespace llvm;
using namespace std;

bool AMDGCNNumCacheLines::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t TtraceCounterInt = 1;
  for (auto &F : M) {
    if(F.isIntrinsic()) continue;
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
          // Global memory reads
          if (LoadInst* LI = dyn_cast<LoadInst>(I)) {      
              IRBuilder<> Builder(dyn_cast<Instruction>(I));
              Value *Addr = LI->getPointerOperand();
              Value *TtraceCounterIntVal = Builder.getInt32(TtraceCounterInt);
              Value *Op = LI->getPointerOperand()->stripPointerCasts();
              uint32_t AddrSpace =
                  cast<PointerType>(Op->getType())->getAddressSpace();
              //Shared and Constant Address Spaces
              if(AddrSpace == 3 || AddrSpace == 4) continue;
              
                StringRef UnmangledName = getUnmangledName(F.getName());
             
              DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();
//
              std::string SourceInfo =
                  (F.getName() + "     " + DL->getFilename() + ":" +
                   Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                      .str();

              Type* dataType = LI->getType();
	      	  const DataLayout dl = M.getDataLayout();
              uint32_t typeSize = dl.getTypeStoreSize(dataType); 
              Value *typeSizeVal = Builder.getInt32(typeSize);
              FunctionType *FT = FunctionType::get(Type::getVoidTy(CTX), {Addr->getType(), Type::getInt32Ty(CTX), Type::getInt32Ty(CTX)}, false);
              FunctionCallee InstrumentationFunction = M.getOrInsertFunction("_Z13numCacheLinesPvjj", FT);
			  Builder.CreateCall(InstrumentationFunction, {Addr, TtraceCounterIntVal, typeSizeVal});
              errs() << "Injecting Num Cache Lines Function Into AMDGPU Kernel: " << UnmangledName
                     << "\n";                     
              errs() << TtraceCounterInt << "     " << SourceInfo << "\n";
              TtraceCounterInt++;   
              ModifiedCodeGen = true;                                                                     
          }
        }
      }
  } 
}
    return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
        MPM.addPass(AMDGCNNumCacheLines());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "amdgcn-num-cache-lines",
          LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
