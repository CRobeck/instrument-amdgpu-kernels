#include "InjectAMDGCNSharedMemTtrace.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <iostream>
using namespace llvm;

static cl::opt<std::string> InstrumentAMDGPUFunction("instrument-amdgpu-function", cl::init(""),
                          cl::desc("AMDGPU function to instrument"));

static GlobalVariable *addGlobalArray(unsigned NumElts, llvm::Type *ElemType,
                                      unsigned int AddrSpace,
                                      llvm::Module *mainModule,
                                      std::string name) {
      ArrayType *ArrayTy = ArrayType::get(ElemType, NumElts);
      GlobalVariable *GlobalArray = new GlobalVariable(
          /*Module=*/*mainModule,
          /*Type=*/ArrayTy,
          /*isConstant=*/false,
          /*Linkage=*/GlobalValue::InternalLinkage,
          /*Initializer=*/nullptr, // has initializer, specified below
          /*Name=*/name.c_str(),
          /*InsertBefore*/ nullptr, GlobalVariable::NotThreadLocal, AddrSpace);
      std::vector<llvm::Constant *> ArrayVals;
      for (int i = 0; i < NumElts; i++)
        ArrayVals.push_back(ConstantInt::get(ElemType, 0));
  GlobalArray->setInitializer(ConstantArray::get(ArrayTy, ArrayVals));
  return GlobalArray;
}

template <typename LoadOrStoreInst>
bool checkInstTypeAndAddressSpace(BasicBlock::iterator &I, Function &F,
                                  unsigned CounterInt,
                                  bool &DebugInfoWarningPrinted) {
  auto LSI = dyn_cast<LoadOrStoreInst>(I);
  if (not LSI) {
    return false;
  }

  Value *Op = LSI->getPointerOperand()->stripPointerCasts();
  unsigned AddrSpace = cast<PointerType>(Op->getType())->getAddressSpace();
  if (AddrSpace != 3) {
    return false;
  }

  if (DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc()) {
    std::string SourceInfo =
        (F.getName() + "\t" + DL->getFilename() + ":" + Twine(DL->getLine()) +
         ":" + Twine(DL->getColumn()))
            .str();
    errs() << CounterInt << "\t" << SourceInfo << "\n";
  } else {
    if (!DebugInfoWarningPrinted) {
      errs() << "warning: no debug info found, did you forget to "
                "add -ggdb?\n";
      DebugInfoWarningPrinted = true;
    }
  }

  return true;
}

Value* initFlagPtr(Function& F, LLVMContext &CTX,
                          GlobalVariable* GlobalAtomicFlagsArray){
  auto BB = F.begin();
  if(BB == F.end()){ return nullptr; }

  auto I = BB->begin();
  if(I == BB->end()){ return nullptr; }

  IRBuilder<> Builder(dyn_cast<Instruction>(I));
  Builder.SetInsertPoint(dyn_cast<Instruction>(I));
  StructType *STy = StructType::get(CTX,{Type::getInt32Ty(CTX),
                                         Type::getInt32Ty(CTX),
                                         Type::getInt32Ty(CTX)});
  FunctionType *FTy = FunctionType::get(STy, false);
  // Compute 9-bits offset for this wave in GlobalAtomicFlagsArray
  // based on SE | CU | SIMD and store offset into $0
  Value* S =Builder.CreateCall(InlineAsm::get(FTy,
                                    "s_getreg_b32 $1, hwreg(HW_REG_HW_ID)\n"
                                    "s_bfe_u32 $0, $1, 0x3000d\n" // SE id: 3 bits, 13-15
                                    "s_lshl_b32 $0, $0, 6\n"       // create space for CU | SIMD
                                    "s_bfe_u32 $2, $1, 0x40008\n" // CU id: 4 bits, 8-11
                                    "s_lshl_b32 $2, $2, 2\n"         // create space for SIMD
                                    "s_or_b32 $0, $0, $2\n"       // combine SE | CU
                                    "s_bfe_u32 $2, $1, 0x20004\n" // SIMD id: 2 bits, 4-5
                                    "s_or_b32 $0, $0, $2\n"         // combine SE | CU | SIMD
                                    , "=s,=s,=s", true),{});
  Value* FlagOffset = Builder.CreateExtractValue(S, 0);
  FlagOffset = Builder.CreateZExt(FlagOffset, Type::getInt64Ty(CTX));
  Value* FlagPtr = Builder.CreateInBoundsGEP(Type::getInt32Ty(CTX), GlobalAtomicFlagsArray, FlagOffset);

  // At this point, FlagPtr is a pointer into an address space (space 1, or global device memory,
  // in our case). Such a pointer can be used for load and store ops, but if we use it with
  // s_atomic_cmpswap, it will trigger a linker error "Invalid record". To fix that, we need to
  // cast the address space away.
  FlagPtr = Builder.CreateAddrSpaceCast(FlagPtr, Type::getInt32PtrTy(CTX));
  return FlagPtr;
}

template <typename LoadOrStoreInst>
void instrumentIfLDSInstruction(BasicBlock::iterator &I, LLVMContext &CTX,
                                Function &F, unsigned &CounterInt,
                                Value *&TtraceCounter,
                                bool &DebugInfoWarningPrinted) {
  if (not checkInstTypeAndAddressSpace<LoadOrStoreInst>(
          I, F, CounterInt, DebugInfoWarningPrinted)) {
    return;
  }

  IRBuilder<> Builder(dyn_cast<Instruction>(I));
  Builder.SetInsertPoint(dyn_cast<Instruction>(I));
  FunctionType *FTy =
      FunctionType::get(Type::getInt32Ty(CTX), {Type::getInt32Ty(CTX)}, false);
  Value *OldM0 = Builder.CreateCall(InlineAsm::get(FTy,
                                                   "s_mov_b32 $0 m0\n"
                                                   "s_mov_b32 m0 $1\n"
                                                   "s_nop 0\n",
                                                   "=s,s", true),
                                    {TtraceCounter});
  Builder.SetInsertPoint(dyn_cast<Instruction>(std::next(I, 1)));
  FunctionType *FTy2 =
      FunctionType::get(Type::getInt32Ty(CTX),
                        {Type::getInt32Ty(CTX), Type::getInt32Ty(CTX)}, false);
  TtraceCounter = Builder.CreateCall(InlineAsm::get(FTy2,
                                                    "s_ttracedata\n"
                                                    "s_mov_b32 m0 $1\n"
                                                    "s_add_i32 $0 $2 1\n"
                                                    ".rept 16\n"
                                                    "s_nop 15\n"
                                                    ".endr\n",
                                                    "=s,s,s", true),
                                     {OldM0, TtraceCounter});
  CounterInt++;
}

void printIR(Function &F) {
  errs() << F.getName() << '\n';
  for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
    for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
      errs() << *I << "\n";
    }
  }
}

bool InjectAMDGCNSharedMemTtrace::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  bool DebugInfoWarningPrinted = false;
  IRBuilder<> ModuleBuilder(CTX);
  // This is the actual variable value that gets inserted in the Inline ASM
  Value *TtraceCounter = ModuleBuilder.getInt32(0);
  // This is the internal counter in the compiler pass. These two will not match
  // currently b/c unrolled loops will copy/inline the InlineASM version not the
  // internal compiler counter.
  unsigned CounterInt = 0;
  for (auto &F : M) {
    if (F.getCallingConv() == CallingConv::AMDGPU_KERNEL) {
      if (F.getName() == InstrumentAMDGPUFunction ||
          InstrumentAMDGPUFunction.empty()) {
        bool print_ir = false;
        if(print_ir){ printIR(F); }
        GlobalVariable *GlobalAtomicFlagsArray =
            addGlobalArray(512, Type::getInt32Ty(CTX), 1, &M, "atomicFlags");
        Value* FlagPtr = initFlagPtr(F, CTX, GlobalAtomicFlagsArray);
        for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
          for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
            // Shared memory reads
            instrumentIfLDSInstruction<LoadInst>(
                I, CTX, F, CounterInt, TtraceCounter, DebugInfoWarningPrinted);
            // Shared memory writes
            instrumentIfLDSInstruction<StoreInst>(
                I, CTX, F, CounterInt, TtraceCounter, DebugInfoWarningPrinted);
          }
        } // End of instructions in AMDGCN kernel loop

        errs() << "Injected LDS Load/Store s_ttrace instructions at "
               << CounterInt << " source locations\n";

        ModifiedCodeGen = true;
        if(print_ir){
          errs() << "After instrumentation:\n";
          printIR(F);
        }
      }
    } // End of if AMDGCN Kernel
  }   // End of functions in module loop
  return ModifiedCodeGen;
}

PassPluginLibraryInfo getPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
      MPM.addPass(InjectAMDGCNSharedMemTtrace());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "inject-amdgcn-lds-ttrace",
          LLVM_VERSION_STRING, callback};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
