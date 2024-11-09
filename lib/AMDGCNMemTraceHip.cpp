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

std::map<int, std::string> AddrSpaceMap = {
    {0, "FLAT"}, {1, "GLOBAL"}, {3, "SHARED"}, {4, "CONSTANT"}};

std::string LoadOrStoreMap(const BasicBlock::iterator &I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return "LOAD";
  else if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return "STORE";
  else
    throw std::runtime_error("Error: unknown operation type");
}

template <typename LoadOrStoreInst>
void InjectInstrumentationFunction(const BasicBlock::iterator &I,
                                   const Function &F, llvm::Module &M,
                                   uint32_t &LocationCounter, llvm::Value *Ptr,
                                   bool PrintLocationInfo) {
  auto &CTX = M.getContext();
  auto LSI = dyn_cast<LoadOrStoreInst>(I);
  Value *AccessTypeVal;
  Type *PointeeType;
  IRBuilder<> Builder(dyn_cast<Instruction>(I));
  auto LI = dyn_cast<LoadInst>(I);
  auto SI = dyn_cast<StoreInst>(I);
  if (LI) {
    AccessTypeVal = Builder.getInt8(0b01);
    PointeeType = LI->getType();
  } else if (SI) {
    AccessTypeVal = Builder.getInt8(0b10);
    PointeeType = SI->getValueOperand()->getType();
  } else {
    return;
  }  
  if (not LSI)
    return;
  
  Value *Addr = LSI->getPointerOperand();
  Value *LocationCounterVal = Builder.getInt32(LocationCounter);
  Value *Op = LSI->getPointerOperand()->stripPointerCasts();
  uint32_t AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
  Value *AddrSpaceVal = Builder.getInt8(AddrSpace);
  uint16_t PointeeTypeSize = M.getDataLayout().getTypeStoreSize(PointeeType);
  Value *PointeeTypeSizeVal = Builder.getInt16(PointeeTypeSize);  
  // Skip shared and constant memory address spaces for now
  if (AddrSpace == 3 || AddrSpace == 4)
    return;
  DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

  std::string SourceInfo = (F.getName() + "     " + DL->getFilename() + ":" +
                            Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                               .str();

  FunctionType *FT =  FunctionType::get(Type::getVoidTy(CTX),
                        {Ptr->getType(), Addr->getType(), Type::getInt32Ty(CTX),
                         Type::getInt8Ty(CTX), Type::getInt8Ty(CTX),
                         Type::getInt16Ty(CTX)},
                        false);                                       
  FunctionCallee InstrumentationFunction =
          M.getOrInsertFunction("v_submit_address", FT);   
  Builder.CreateCall(FT, cast<Function>(InstrumentationFunction.getCallee()), 
  {Ptr, Addr, LocationCounterVal, AccessTypeVal, AddrSpaceVal,
       PointeeTypeSizeVal});
  if (PrintLocationInfo) {
    errs() << "Injecting Mem Trace Function Into AMDGPU Kernel: " << SourceInfo
           << "\n";
    errs() << LocationCounter << "     " << SourceInfo << "     "
           << AddrSpaceMap[AddrSpace] << "     " << LoadOrStoreMap(I) << "\n";
  }
  LocationCounter++;
}

bool AMDGCNNumCacheLines::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
//  std::string errorMsg;
//  std::unique_ptr<llvm::Module> InstrumentationModule;
  std::vector<Function *> GpuKernels;
//  if (!loadInstrumentationFile(InstrumentationFunctionFile, CTX,
//                               InstrumentationModule, errorMsg)) {
//    printf("error loading program '%s': %s",
//           InstrumentationFunctionFile.c_str(), errorMsg.c_str());
//    exit(1);
//  }
//  Linker::linkModules(M, std::move(InstrumentationModule));
  // Get the unmodified kernels first so we don't end up in a infinite loop
  for (auto &F : M) {
    if (F.isIntrinsic())
      continue;
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      GpuKernels.push_back(&F);
    }
  }
  for (auto &I : GpuKernels) {
    std::string AugmentedName = "__amd_crk_" + I->getName().str() + "Pv";
    ValueToValueMapTy VMap;
    // Add an extra ptr arg on to the instrumented kernels
    std::vector<Type *> ArgTypes;
    for (auto arg = I->arg_begin(); arg != I->arg_end(); ++arg) {
      ArgTypes.push_back(arg->getType());
    }
    ArgTypes.push_back(PointerType::get(M.getContext(), /*AddrSpace=*/0));
    FunctionType *FTy =
        FunctionType::get(I->getFunctionType()->getReturnType(), ArgTypes,
                          I->getFunctionType()->isVarArg());
    Function *NF = Function::Create(FTy, I->getLinkage(), I->getAddressSpace(),
                                    AugmentedName, &M);
    NF->copyAttributesFrom(I);
    VMap[I] = NF;

    // Get the ptr we just added to the kernel arguments
    Value *bufferPtr = &*NF->arg_end() - 1;
    Function *F = cast<Function>(VMap[I]);

    Function::arg_iterator DestI = F->arg_begin();
    for (const Argument &J : I->args()) {
      DestI->setName(J.getName());
      VMap[&J] = &*DestI++;
    }
    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(F, I, VMap, CloneFunctionChangeType::GlobalChanges,
                      Returns);
    uint32_t LocationCounter = 0;
    for (Function::iterator BB = NF->begin(); BB != NF->end(); BB++) {
      for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
        if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
          InjectInstrumentationFunction<LoadInst>(I, *NF, M, LocationCounter,
                                                  bufferPtr, true);
          ModifiedCodeGen = true;
        } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
          InjectInstrumentationFunction<StoreInst>(I, *NF, M, LocationCounter,
                                                   bufferPtr, true);
          ModifiedCodeGen = true;
        }
      }
    }
  }
  return ModifiedCodeGen;
}

//bool AMDGCNNumCacheLines::runOnModule(Module &M) {
//  bool ModifiedCodeGen = false;
//  auto &CTX = M.getContext();
//  uint32_t TtraceCounterInt = 1;
//  for (auto &F : M) {
//    if(F.isIntrinsic()) continue;
//    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
//      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
//        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
//          // Global memory reads
//          if (LoadInst* LI = dyn_cast<LoadInst>(I)) {      
//              IRBuilder<> Builder(dyn_cast<Instruction>(I));
//              Value *Addr = LI->getPointerOperand();
//              Value *TtraceCounterIntVal = Builder.getInt32(TtraceCounterInt);
//              Value *Op = LI->getPointerOperand()->stripPointerCasts();
//              uint32_t AddrSpace =
//                  cast<PointerType>(Op->getType())->getAddressSpace();
//              //Shared and Constant Address Spaces
//              if(AddrSpace == 3 || AddrSpace == 4) continue;
//              
//                StringRef UnmangledName = getUnmangledName(F.getName());
//             
//              DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();
////
//              std::string SourceInfo =
//                  (F.getName() + "     " + DL->getFilename() + ":" +
//                   Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
//                      .str();
//
//              Type* dataType = LI->getType();
//	      	  const DataLayout dl = M.getDataLayout();
//              uint32_t typeSize = dl.getTypeStoreSize(dataType); 
//              Value *typeSizeVal = Builder.getInt32(typeSize);
//              FunctionType *FT = FunctionType::get(Type::getVoidTy(CTX), {Addr->getType(), Type::getInt32Ty(CTX), Type::getInt32Ty(CTX)}, false);
//              //FunctionCallee InstrumentationFunction = M.getOrInsertFunction("_Z13numCacheLinesPvjj", FT);
//			  //Builder.CreateCall(InstrumentationFunction, {Addr, TtraceCounterIntVal, typeSizeVal});
//              errs() << "Injecting Num Cache Lines Function Into AMDGPU Kernel: " << UnmangledName
//                     << "\n";                     
//              errs() << TtraceCounterInt << "     " << SourceInfo << "\n";
//              TtraceCounterInt++;   
//              ModifiedCodeGen = true;                                                                     
//          }
//        }
//      }
//  } 
//}
//    return ModifiedCodeGen;
//}

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
