#include "AMDGCNMemCoalescing.h"
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


std::string InstrumentationFunctionFile = instrumentation::utils::getenv("AMDCGN_INSTRUMENTATION_FILE");
std::string InstrumentationFunctionName = instrumentation::utils::getenv("AMDCGN_INSTRUMENTATION_FUNCTION");
bool AMDGCNMemCoalescing::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t TtraceCounterInt = 1;
//  const std::string &module_triple = M.getTargetTriple();
//   if(module_triple != "amdgcn-amd-amdhsa") return ModifiedCodeGen;
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
              IRBuilder<> Builder(dyn_cast<Instruction>(I));
              Value *Addr = LI->getPointerOperand();
              Value *TtraceCounterIntVal = Builder.getInt32(TtraceCounterInt);
              Value *Op = LI->getPointerOperand()->stripPointerCasts();
              uint32_t AddrSpace =
                  cast<PointerType>(Op->getType())->getAddressSpace();
              //Shared and Constant Address Spaces
              if(AddrSpace == 3 || AddrSpace == 4) continue;
              
                StringRef UnmangledName = getUnmangledName(F.getName());
             
              SmallVector<StringRef, 10> BuiltinArgsTypeStrs;
              std::string DemangledCall = demangle(std::string(InstrumentationFunctionName));
               
             StringRef DemangledCallStringRef = StringRef(DemangledCall);
              StringRef BuiltinArgs =
                  DemangledCallStringRef.slice(DemangledCallStringRef.find('(') + 1, DemangledCallStringRef.find(')'));   
              BuiltinArgs.split(BuiltinArgsTypeStrs, ',', -1, false);
              SmallVector<Type*, 10> ArgTypes;
              for(int ArgIdx=0;ArgIdx<BuiltinArgsTypeStrs.size();ArgIdx++){
                StringRef TypeStr = BuiltinArgsTypeStrs[ArgIdx].trim();   
                if(TypeStr.ends_with("*")){
                  ArgTypes.push_back(PointerType::get(CTX, 1));
                }
                else {
                    Type* BaseType = parseBasicTypeName(TypeStr, CTX);
                    ArgTypes.push_back(BaseType);
                }
              }

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
              Function *InstrumentationFunction = M.getFunction("_Z13numCacheLinesPvjj");
              Builder.CreateCall(FunctionType::get(Type::getVoidTy(CTX), {Addr->getType(), Type::getInt32Ty(CTX), Type::getInt32Ty(CTX)} ,false), InstrumentationFunction, {Addr, TtraceCounterIntVal, typeSizeVal});
//
//              Function *PrintFunction = M.getFunction("_Z15PrintCacheLinesj");
//              Builder.CreateCall(FunctionType::get(Type::getVoidTy(CTX), {Type::getInt32Ty(CTX)}, false), PrintFunction, {Builder.getInt32(1)});
              errs() << "Injecting Mem Coalescing Function Into AMDGPU Kernel: " << UnmangledName
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
