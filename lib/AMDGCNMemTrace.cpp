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
#include <sstream>
#include <iomanip> 
#include <map>

using namespace llvm;
using namespace std;

std::string InstrumentationFunctionFile = instrumentation::utils::getenv("AMDCGN_INSTRUMENTATION_FUNCTIONS_FILE");
std::map<int, std::string> AddrSpaceMap = {{0, "FLAT"},
					   {1, "GLOBAL"},
					   {3, "SHARED"},
					   {4, "CONSTANT"}};

std::map<std::string, uint32_t> LocationCounterSourceMap;

std::string LoadOrStoreMap(const BasicBlock::iterator &I){
		if (LoadInst* LI = dyn_cast<LoadInst>(I)) return "LOAD";
		else if (StoreInst* SI = dyn_cast<StoreInst>(I)) return "STORE";
		else throw std::runtime_error("Error: unknown operation type");
}
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
	//Skip shared and constant memory address spaces for now
	if(AddrSpace == 3 || AddrSpace == 4) return;
	DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

        std::string SourceAndAddrSpaceInfo =
            (F.getName() + "     " + DL->getFilename() + ":" +
             Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                .str() + "     " + AddrSpaceMap[AddrSpace] + "     " + LoadOrStoreMap(I);

	if(LocationCounterSourceMap.find(SourceAndAddrSpaceInfo) == LocationCounterSourceMap.end()){
		errs() << LocationCounter << "     " << SourceAndAddrSpaceInfo << "\n";
		LocationCounterSourceMap[SourceAndAddrSpaceInfo]=LocationCounter;
		LocationCounter++;
	}
	
        Function *InstrumentationFunction = M.getFunction("_Z8memTracePvj");
        Builder.CreateCall(FunctionType::get(Type::getVoidTy(CTX), {Addr->getType(), Type::getInt32Ty(CTX)} ,false), InstrumentationFunction, {Addr, Builder.getInt32(LocationCounterSourceMap[SourceAndAddrSpaceInfo])});
}



bool AMDGCNMemTrace::runOnModule(Module &M) {
  printf("Hello from AMDGCNMemTrace.cpp, runModule()\n");
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t LocationCounter = 0;
  std::string errorMsg;
  std::unique_ptr<llvm::Module> InstrumentationModule;
  printf("Calling loadInstrumentationFile() with file %s\n", InstrumentationFunctionFile.c_str());
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
          if (LoadInst* LI = dyn_cast<LoadInst>(I)) {      
	      InjectingInstrumentationFunction<LoadInst>(I, F, M, LocationCounter);
              ModifiedCodeGen = true;                                                                     
          }
	  else if(StoreInst* SI = dyn_cast<StoreInst>(I)){
		  InjectingInstrumentationFunction<StoreInst>(I, F, M, LocationCounter);
		  ModifiedCodeGen = true;
	  }
        }
      }
  } 
}
    return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  printf("Hello from AMDGCNMemTrace.cpp, getPassPluginInfo()\n");
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto&&... args) {
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
