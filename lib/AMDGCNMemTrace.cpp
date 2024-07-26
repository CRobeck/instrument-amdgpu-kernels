#include "AMDGCNMemTrace.h"
#include "utils.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <vector>
using namespace llvm;
using namespace std;

template <typename LoadOrStoreInst> 
void InjectingInstrumentationFunction(const BasicBlock::iterator &I, const Function &F, const llvm::Module &M,
				      uint32_t &LocationCounter){
	auto &CTX = M.getContext();
	auto LSI = dyn_cast<LoadOrStoreInst>(I);
	if (not LSI) return;
	IRBuilder<> Builder(dyn_cast<Instruction>(I));
	Value *Addr = LSI->getPointerOperand();
	Value *Op = LSI->getPointerOperand()->stripPointerCasts();
        uint32_t AddrSpace =
                  cast<PointerType>(Op->getType())->getAddressSpace();
	//Shared and Constant Address Spaces
//	if(AddrSpace == 3 || AddrSpace == 4) return;
	DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

        std::string SourceInfo =
            (F.getName() + "     " + DL->getFilename() + ":" +
             Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                .str();	
	
        Function *InstrumentationFunction = M.getFunction("_Z8memTracePv");
        Builder.CreateCall(FunctionType::get(Type::getVoidTy(CTX), {Addr->getType()} ,false), InstrumentationFunction, {Addr});
        errs() << "Injecting Mem Trace Function Into AMDGPU Kernel: " << SourceInfo
               << "\n";
        errs() << LocationCounter << "     " << SourceInfo << "\n";	
	LocationCounter++;
}


std::string InstrumentationFunctionFile = instrumentation::utils::getenv("AMDCGN_INSTRUMENTATION_FUNCTIONS_FILE");
bool AMDGCNMemTrace::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t TtraceCounterInt = 0;
  std::string errorMsg;
  std::unique_ptr<llvm::Module> InstrumentationModule;
  if (!loadInstrumentationFile(InstrumentationFunctionFile, CTX, InstrumentationModule, errorMsg)) {
    printf("error loading program '%s': %s", InstrumentationFunctionFile.c_str(),
               errorMsg.c_str());
    exit(1);
  }
  Linker::linkModules(M, std::move(InstrumentationModule));  
  for (auto &F : M) {
    if(F.isIntrinsic()) continue;
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
          // Global memory reads
          if (LoadInst* LI = dyn_cast<LoadInst>(I)) {      
	      InjectingInstrumentationFunction<LoadInst>(I, F, M, TtraceCounterInt);
              ModifiedCodeGen = true;                                                                     
          }
	  else if(StoreInst* SI = dyn_cast<StoreInst>(I)){
		  InjectingInstrumentationFunction<StoreInst>(I, F, M, TtraceCounterInt);
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
        MPM.addPass(AMDGCNMemTrace());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "amdgcn-mem-trace",
          LLVM_VERSION_STRING, callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
