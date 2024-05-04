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
#include "llvm/Demangle/Demangle.h"
#include "llvm/Demangle/ItaniumDemangle.h"
#include "llvm/Support/Allocator.h"
#include <iostream>
#include <vector>
using namespace llvm;
using namespace std;


static cl::list<std::string>
    KernelsToInstrument("amdgcn-kernels-to-instrument",
                       cl::desc("Specify function(s) to instrument using a "
                                "regular expression"));

//static cl::opt<std::string> InstrumentationFunctionName("amdgcn-instrumentation-function",
//                       cl::desc("Specify function to inject"));      

static cl::list<std::string> InstrumentationPoint("amdgcn-instrumentation-point",
                       cl::desc("Specify point in function inject instrumentation function")); 
static std::string InstrumentationFunctionName = "_Z13numCacheLinesPvjjj";

bool AMDGCNMemCoalescing::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t TtraceCounterInt = 1;    
  for (auto &F : M) {
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
                  // errs() << "Pointer" << '\n'; 
                  ArgTypes.push_back(PointerType::get(CTX, 1));
                }
                else {
                    // TypeStr = TypeStr.slice(0, TypeStr.find_first_of(" *"));
                    Type* BaseType = parseBasicTypeName(TypeStr, CTX);
                    ArgTypes.push_back(BaseType);
                    // errs() << *BaseType << '\n';     
                }
              }

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
              // FunctionType *FT = FunctionType::get(Type::getInt32Ty(CTX),
              //                                      {PointerType::get(CTX, 1),
              //                                      Type::getInt32Ty(CTX),
              //                                      Type::getInt32Ty(CTX), Type::getInt32Ty(CTX)}, false);                                            
              FunctionCallee InstrumentationFunctionCallee =
                  M.getOrInsertFunction(InstrumentationFunctionName, FunctionType::get(Type::getInt32Ty(CTX),ArgTypes,false));
              FunctionCallee PrintFunctionCallee =
                  M.getOrInsertFunction("_Z15PrintCacheLinesj", FunctionType::get(Type::getVoidTy(CTX), {Type::getInt32Ty(CTX)}, false));     
              Function *PrintFunction =
                  cast<Function>(PrintFunctionCallee.getCallee());                                 
              Value* NumCacheLines = Builder.CreateCall(InstrumentationFunctionCallee, {Addr, loadVal, TtraceCounterIntVal, typeSizeVal});
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
