#include "AMDGCNMemCoalescing.h"
#include "utils.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <iostream>
#include <vector>
using namespace llvm;
using namespace std;


static cl::list<std::string>
    AMDGCNKernelsToInstrument("amdgcn-kernels-to-instrument",
                       cl::desc("Specify function(s) to instrument using a "
                                "regular expression"));

static cl::opt<std::string> AMDGCNInstrumentationFunction("amdgcn-instrumentation-function",
                       cl::desc("Specify function to inject"));      

static cl::opt<std::string> AMDGCNInstrumentationPoint("amdgcn-instrumentation-point",
                       cl::desc("Specify point in function inject instrumentation-function"));                         

bool AMDGCNMemCoalescing::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t TtraceCounterInt = 1;    
  for (auto &F : M) {
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
          // Shared memory reads
          if (LoadInst* LI = dyn_cast<LoadInst>(I)) {      
              IRBuilder<> Builder(dyn_cast<Instruction>(I));
              Value *Addr = LI->getPointerOperand();
              Value *TtraceCounterIntVal = Builder.getInt32(TtraceCounterInt);
              Value *Op = LI->getPointerOperand()->stripPointerCasts();
              uint32_t AddrSpace =
                  cast<PointerType>(Op->getType())->getAddressSpace();
              Value *GEP = dyn_cast<GetElementPtrInst>(Op)->getPointerOperand()->stripPointerCasts(); 
			        //Shared and Constant Address Spaces
              if(AddrSpace == 3 || AddrSpace == 4) continue;
              
              StringRef UnmangledName = getUnmangledName(F.getName());
              // Value* storeVal = Builder.CreateGlobalStringPtr("Load");
              Value* loadVal = Builder.getInt32(1); //0=Store, 1=Load

              DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

              std::string SourceInfo =
                  (F.getName() + "     " + DL->getFilename() + ":" +
                   Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                      .str();
              errs() << TtraceCounterInt << "     " << SourceInfo << "\n";

              Type* dataType = LI->getType();
              DataLayout* dl = new DataLayout(&M);
              uint32_t typeSize = dl->getTypeStoreSize(dataType); 
              Value *typeSizeVal = Builder.getInt32(typeSize);

              FunctionType *FT = FunctionType::get(Type::getInt32Ty(CTX),
                                                   {cast<PointerType>(GEP->getType()),
                                                   Type::getInt32Ty(CTX),
                                                   Type::getInt32Ty(CTX), Type::getInt32Ty(CTX)}, false);                               
              FunctionCallee InjectedFunctionCallee =
                  M.getOrInsertFunction("_Z15countCacheLinesPvjjj", FT);
              FunctionCallee PrintFunctionCallee =
                  M.getOrInsertFunction("_Z15PrintCacheLinesj", FunctionType::get(Type::getVoidTy(CTX), {Type::getInt32Ty(CTX)}, false));     
              Function *PrintFunction =
                  cast<Function>(PrintFunctionCallee.getCallee());                                 
              Value* NumCacheLines = Builder.CreateCall(InjectedFunctionCallee, {Addr, loadVal, TtraceCounterIntVal, typeSizeVal});
              errs() << "Injecting Mem Coalescing Function Into AMDGPU Kernel: " << UnmangledName
                     << "\n";      
              Builder.CreateCall(PrintFunction, {NumCacheLines});                   
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
        MPM.addPass(AMDGCNMemCoalescing());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "amdgcn-mem-coalescing",
          LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
