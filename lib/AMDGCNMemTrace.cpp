#include "AMDGCNMemTrace.h"
#include "utils.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
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
		IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
		if(II && II->getIntrinsicID() == Intrinsic::masked_load) return "LOAD";
		else if(II && II->getIntrinsicID() == Intrinsic::masked_store) return "STORE";
		else if (LoadInst* LI = dyn_cast<LoadInst>(I)) return "LOAD";
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

//void instrumentAddress(Instruction *OrigIns,
//                                    Instruction *InsertBefore, Value *Addr, Type *IntptrTy){
//  IRBuilder<> IRB(InsertBefore);
//  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
//  Function *InstrumentationFunction = M.getFunction("_Z8memTracePvj");
//  Builder.CreateCall(FunctionType::get(Type::getVoidTy(CTX), {IntptrTy, Type::getInt32Ty(CTX)} ,false), InstrumentationFunction, {AddrLong, Builder.getInt32(LocationCounterSourceMap[SourceAndAddrSpaceInfo])});
//
//}

void instrumentMaskedLoadOrStore(const DataLayout &DL, Value *Mask,
                                              Instruction *I, Value *Addr,
                                              Type *AccessTy, const llvm::Module &M, bool IsWrite) {
  auto *VTy = cast<FixedVectorType>(AccessTy);
  unsigned Num = VTy->getNumElements();
  int LongSize = DL.getPointerSizeInBits();
  LLVMContext *CTX = &(M.getContext());
  Type *IntptrTy = Type::getIntNTy(*CTX, LongSize);	
  auto *Zero = ConstantInt::get(IntptrTy, 0);
  for (unsigned Idx = 0; Idx < Num; ++Idx) {
    Value *InstrumentedAddress = nullptr;
    Instruction *InsertBefore = I;
    if (auto *Vector = dyn_cast<ConstantVector>(Mask)) {
      // dyn_cast as we might get UndefValue
      if (auto *Masked = dyn_cast<ConstantInt>(Vector->getOperand(Idx))) {
        if (Masked->isZero())
          // Mask is constant false, so no instrumentation needed.
          continue;
        // If we have a true or undef value, fall through to instrumentAddress.
        // with InsertBefore == I
      }
    } else {
      IRBuilder<> IRB(I);
//      errs() << *I << "\n";
      Value *MaskElem = IRB.CreateExtractElement(Mask, Idx);
      Instruction *ThenTerm = SplitBlockAndInsertIfThen(MaskElem, I, false);
//      errs() << *ThenTerm << "\n";
      InsertBefore = ThenTerm; 
    }

    IRBuilder<> IRB(InsertBefore);
    InstrumentedAddress =
        IRB.CreateGEP(VTy, Addr, {Zero, ConstantInt::get(IntptrTy, Idx)});
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  //Function *InstrumentationFunction = M.getFunction("_Z8memTracePvj");
  //IRB.CreateCall(FunctionType::get(Type::getVoidTy(CTX), {IntptrTy, Type::getInt32Ty(CTX)} ,false), InstrumentationFunction, {AddrLong, IRB.getInt32(LocationCounterSourceMap[SourceAndAddrSpaceInfo])});
  }
}

template <>
void InjectingInstrumentationFunction<IntrinsicInst>(const BasicBlock::iterator &I, const Function &F, const llvm::Module &M,
                                      uint32_t &LocationCounter){
 	LLVMContext *CTX = &(M.getContext());
        auto *CI = dyn_cast<CallInst>(I);
        if (not CI) return;	
	unsigned OpOffset = 0;
	Type *AccessTy;
	IntrinsicInst *II = dyn_cast<IntrinsicInst>(I);
	bool IsWrite = false;
	if(II && II->getIntrinsicID() == Intrinsic::masked_store){
		OpOffset = 1;
		AccessTy = CI->getArgOperand(0)->getType();
		IsWrite = true;
	}
	if(II && II->getIntrinsicID() == Intrinsic::masked_load){
		OpOffset = 0;
		AccessTy = CI->getType();
	}

	Value *BasePtr = CI->getOperand(0 + OpOffset);
	Value* Mask = CI->getOperand(2 + OpOffset);
	Value *Addr = BasePtr->stripInBoundsOffsets();
	Type *PtrTy = cast<PointerType>(Addr->getType()->getScalarType());
	uint32_t AddrSpace = PtrTy->getPointerAddressSpace();
        //Skip shared and constant memory address spaces for now
        if(AddrSpace == 3 || AddrSpace == 4) return;	

	const DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

        std::string SourceAndAddrSpaceInfo =
            (F.getName() + "     " + DL->getFilename() + ":" +
             Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                .str() + "     " + AddrSpaceMap[AddrSpace] + "     " + LoadOrStoreMap(I);

	if(LocationCounterSourceMap.find(SourceAndAddrSpaceInfo) == LocationCounterSourceMap.end()){
		errs() << LocationCounter << "     " << SourceAndAddrSpaceInfo << "\n";
		LocationCounterSourceMap[SourceAndAddrSpaceInfo]=LocationCounter;
		LocationCounter++;
	}
	instrumentMaskedLoadOrStore(M.getDataLayout(), Mask, cast<Instruction>(std::next(I, 1)), Addr, AccessTy, M, IsWrite);
//    	int LongSize = M.getDataLayout().getPointerSizeInBits();
//    	Type *IntptrTy = Type::getIntNTy(*CTX, LongSize);	
//
//  	auto *VTy = cast<FixedVectorType>(AccessTy);
//  	unsigned Num = VTy->getNumElements();
//  	auto *Zero = ConstantInt::get(IntptrTy, 0);	
//  	for (unsigned Idx = 0; Idx < Num; ++Idx) {
//  	  Value *InstrumentedAddress = nullptr;
//  	  Instruction *InsertBefore = cast<Instruction>(I);
//  	  if (auto *Vector = dyn_cast<ConstantVector>(Mask)) {
//  	    if (auto *Masked = dyn_cast<ConstantInt>(Vector->getOperand(Idx))) {
//  	      if (Masked->isZero())
//  	        continue;
//  	    }
//  	  } else {
//  	    IRBuilder<> IRB(cast<Instruction>(I));
//  	    Value *MaskElem = IRB.CreateExtractElement(Mask, Idx);
////	    errs() << *I << "\n";
////	    errs() << MaskElem << "\n";
////  	    Instruction *ThenTerm = SplitBlockAndInsertIfThen(MaskElem, cast<Instruction>(I), false);
////  	    InsertBefore = ThenTerm;
//	    InsertBefore = cast<Instruction>(I);  	    
//  	  }
////
////	
//
//	  IRBuilder<> IRB(InsertBefore);
////	  IRBuilder<> IRB(dyn_cast<Instruction>(I));
// 	  InstrumentedAddress =
// 	      IRB.CreateGEP(VTy, Addr, {Zero, ConstantInt::get(IntptrTy, Idx)});
//       Function *InstrumentationFunction = M.getFunction("_Z8memTracePvj");
//        IRB.CreateCall(FunctionType::get(Type::getVoidTy(*CTX), {InstrumentedAddress->getType(), Type::getInt32Ty(*CTX)} ,false), InstrumentationFunction, {InstrumentedAddress, IRB.getInt32(LocationCounterSourceMap[SourceAndAddrSpaceInfo])});
//  	}	

}


bool AMDGCNMemTrace::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t LocationCounter = 0;
  std::string errorMsg;
  std::unique_ptr<llvm::Module> InstrumentationModule;
//  execv("hipcc", "./MemTraceInstrumentationKernel.cpp");
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
      	if(IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)){
              if(II->getIntrinsicID() == Intrinsic::masked_load || II->getIntrinsicID() == Intrinsic::masked_store){
		      InjectingInstrumentationFunction<IntrinsicInst>(I, F, M, LocationCounter);

      	}   
	}
	else if (LoadInst* LI = dyn_cast<LoadInst>(I)) {      
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
