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
      if(F.getName() == InstrumentAMDGPUFunction || InstrumentAMDGPUFunction.empty()){
      		GlobalVariable *GlobalAtomicFlagsVar = addGlobalArray(512, Type::getInt32Ty(CTX), 1, &M, "atomicFlags");
      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
          // Shared memory reads
          if (auto LI = dyn_cast<LoadInst>(I)) {
            Value *Op = LI->getPointerOperand()->stripPointerCasts();
            unsigned AddrSpace =
                cast<PointerType>(Op->getType())->getAddressSpace();
            if (AddrSpace == 3) {
              if (DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc()) {
                std::string SourceInfo =
                    (F.getName() + "\t" + DL->getFilename() + ":" +
                     Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                        .str();
                errs() << CounterInt << "\t" << SourceInfo << "\n";
              } else {
                if (!DebugInfoWarningPrinted) {
                  errs() << "warning: no debug info found, did you forget to "
                            "add -ggdb?\n";
                  DebugInfoWarningPrinted = true;
                }
              }
              IRBuilder<> Builder(dyn_cast<Instruction>(I));
              Builder.SetInsertPoint(dyn_cast<Instruction>(std::next(I,-1)));
              FunctionType *FTy =
                  FunctionType::get(Type::getInt32Ty(CTX), true);
              std::string AsmString = "s_mov_b32 $0 m0\n"
                                      "s_mov_b32 m0 $1\n"
                                      "s_nop 0\n";
              InlineAsm *InlineAsmFunc =
                  InlineAsm::get(FTy, AsmString, "=s,s", true);
              Builder.CreateCall(InlineAsmFunc, {TtraceCounter});
              Builder.SetInsertPoint(dyn_cast<Instruction>(std::next(I,1)));
              Builder.CreateCall(InlineAsm::get(FTy,"s_ttracedata\n""s_mov_b32 m0 $0\n""s_add_i32 $1 $1 1\n"
                                                "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"
                                                "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"
                                                "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"
                                                "s_nop 15\n","=s,s", true),{TtraceCounter});
              CounterInt++;
            }
          }
          // Shared memory writes
          if (auto SI = dyn_cast<StoreInst>(I)) {
            Value *Op = SI->getPointerOperand()->stripPointerCasts();
            unsigned AddrSpace =
                cast<PointerType>(Op->getType())->getAddressSpace();
            if (AddrSpace == 3) {
              if (DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc()) {
                std::string SourceInfo =
                    (F.getName() + "\t" + DL->getFilename() + ":" +
                     Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                        .str();
                errs() << CounterInt << "\t" << SourceInfo << "\n";
              } else {
                if (!DebugInfoWarningPrinted) {
                  errs() << "warning: no debug info found, did you forget to "
                            "add -ggdb?\n";
                  DebugInfoWarningPrinted = true;
                }
              }
              IRBuilder<> Builder(dyn_cast<Instruction>(I));
              Builder.SetInsertPoint(dyn_cast<Instruction>(std::next(I,-1)));
              FunctionType *FTy =
                  FunctionType::get(Type::getInt32Ty(CTX), true);
              std::string AsmString = "s_mov_b32 $0 m0\n"
                                      "s_mov_b32 m0 $1\n"
                                      "s_nop 0\n";
              InlineAsm *InlineAsmFunc =
                  InlineAsm::get(FTy, AsmString, "=s,s", true);
              Builder.CreateCall(InlineAsmFunc, {TtraceCounter});
              Builder.SetInsertPoint(dyn_cast<Instruction>(std::next(I,1)));
              Builder.CreateCall(InlineAsm::get(FTy,"s_ttracedata\n""s_mov_b32 m0 $0\n""s_add_i32 $1 $1 1\n"
                                                "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"
                                                "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"
                                                "s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n""s_nop 15\n"
                                                "s_nop 15\n","=s,s", true),{TtraceCounter});
              CounterInt++;
            }
          }
        }
      } // End of instructions in AMDGCN kernel loop
      
      errs() << "Injected LDS Load/Store s_ttrace instructions at "
             << CounterInt << " source locations\n";

      ModifiedCodeGen = true;
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
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getPassPluginInfo();
}
