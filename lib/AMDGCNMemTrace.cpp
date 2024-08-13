#include "AMDGCNMemTrace.h"
#include "utils.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/FileSystem.h"
#include <iostream>
#include <vector>
#include <map>
#include <system_error>

using namespace llvm;
using namespace std;

std::string InstrumentationFunctionFile = instrumentation::utils::getenv("AMDCGN_INSTRUMENTATION_FUNCTIONS_FILE");

std::map<int, std::string> AddrSpaceMap = {{0, "FLAT"},
					   {1, "GLOBAL"},
					   {3, "SHARED"},
					   {4, "CONSTANT"}};

std::string LoadOrStoreMap(const BasicBlock::iterator &I){
		if (LoadInst* LI = dyn_cast<LoadInst>(I)) return "LOAD";
		else if (StoreInst* SI = dyn_cast<StoreInst>(I)) return "STORE";
		else throw std::runtime_error("Error: unknown operation type");
}

static void copyComdat(GlobalObject *Dst, const GlobalObject *Src) {
  const Comdat *SC = Src->getComdat();
  if (!SC)
    return;
  Comdat *DC = Dst->getParent()->getOrInsertComdat(SC->getName());
  DC->setSelectionKind(SC->getSelectionKind());
  Dst->setComdat(DC);
}

template <typename LoadOrStoreInst> 
void InjectInstrumentationFunction(const BasicBlock::iterator &I, const Function &F, const llvm::Module &M,
				      uint32_t &LocationCounter, llvm::Value* Ptr, bool PrintLocationInfo){
	auto &CTX = M.getContext();
	auto LSI = dyn_cast<LoadOrStoreInst>(I);
	if (not LSI) return;
	IRBuilder<> Builder(dyn_cast<Instruction>(I));
	Value *Addr = LSI->getPointerOperand();
	Value *LocationCounterVal = Builder.getInt32(LocationCounter);
	Value *Op = LSI->getPointerOperand()->stripPointerCasts();
        uint32_t AddrSpace =
                  cast<PointerType>(Op->getType())->getAddressSpace();
	DILocation *DL = dyn_cast<Instruction>(I)->getDebugLoc();

        std::string SourceInfo =
            (F.getName() + "     " + DL->getFilename() + ":" +
             Twine(DL->getLine()) + ":" + Twine(DL->getColumn()))
                .str();	
        Function *InstrumentationFunction = M.getFunction("_Z8memTracePvjS_");
        Builder.CreateCall(FunctionType::get(Type::getVoidTy(CTX), {Addr->getType(), Type::getInt32Ty(CTX),Ptr->getType()} ,false), InstrumentationFunction, {Addr, LocationCounterVal, Ptr});
	if(PrintLocationInfo){
        	errs() << "Injecting Mem Trace Function Into AMDGPU Kernel: " << SourceInfo
        	       << "\n";
        	errs() << LocationCounter << "     " << SourceInfo <<  "     " << AddrSpaceMap[AddrSpace] << "     " << LoadOrStoreMap(I) << "\n";
	}
	LocationCounter++;	
}

std::unique_ptr<Module> CloneModuleAndAddArg(
    const Module &M, ValueToValueMapTy &VMap, const Function &Func, 
    function_ref<bool(const GlobalValue *)> ShouldCloneDefinition) {

  assert(M.isMaterialized() && "Module must be materialized before cloning!");

  // First off, we need to create the new module.
  std::unique_ptr<Module> New =
      std::make_unique<Module>(M.getModuleIdentifier(), M.getContext());
  New->setSourceFileName(M.getSourceFileName());
  New->setDataLayout(M.getDataLayout());
  New->setTargetTriple(M.getTargetTriple());
  New->setModuleInlineAsm(M.getModuleInlineAsm());
  New->IsNewDbgInfoFormat = M.IsNewDbgInfoFormat;

  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (const GlobalVariable &I : M.globals()) {
    GlobalVariable *NewGV = new GlobalVariable(
        *New, I.getValueType(), I.isConstant(), I.getLinkage(),
        (Constant *)nullptr, I.getName(), (GlobalVariable *)nullptr,
        I.getThreadLocalMode(), I.getType()->getAddressSpace());
    NewGV->copyAttributesFrom(&I);
    VMap[&I] = NewGV;
  }

// Loop over the functions in the module, making external functions as before
  for (const Function &I : M) {
    if(I.getName() == Func.getName())
	    continue;
    Function *NF =
        Function::Create(cast<FunctionType>(I.getValueType()), I.getLinkage(),
                         I.getAddressSpace(), I.getName(), New.get());
    NF->copyAttributesFrom(&I);
    VMap[&I] = NF;
  }
  //Add an extra ptr arg on to the instrumented kernels
  std::vector<Type *> ArgTypes;
  for (auto arg = Func.arg_begin(); arg != Func.arg_end();
       ++arg) {
    ArgTypes.push_back(arg->getType());
  }
  ArgTypes.push_back(PointerType::get(New.get()->getContext(), /*AddrSpace=*/0));
  FunctionType *FTy =
    FunctionType::get(Func.getFunctionType()->getReturnType(), ArgTypes,
                      Func.getFunctionType()->isVarArg());    
  Function *NF =
      Function::Create(FTy, Func.getLinkage(),
                       Func.getAddressSpace(), Func.getName(), New.get());
  NF->copyAttributesFrom(&Func);
  VMap[&Func] = NF; 
  
  //Get the ptr we just added to the kernel arguments
  Value *bufferPtr = &*NF->arg_begin() + 4;
// PointerType *VoidPtrType = PointerType::getUnqual(New.get()->getContext());
// Constant *bufferPtr = ConstantPointerNull::get(VoidPtrType);

  // Loop over the aliases in the module
  for (const GlobalAlias &I : M.aliases()) {
    if (!ShouldCloneDefinition(&I)) {
      // An alias cannot act as an external reference, so we need to create
      // either a function or a global variable depending on the value type.
      // FIXME: Once pointee types are gone we can probably pick one or the
      // other.
      GlobalValue *GV;
      if (I.getValueType()->isFunctionTy())
        GV = Function::Create(cast<FunctionType>(I.getValueType()),
                              GlobalValue::ExternalLinkage, I.getAddressSpace(),
                              I.getName(), New.get());
      else
        GV = new GlobalVariable(*New, I.getValueType(), false,
                                GlobalValue::ExternalLinkage, nullptr,
                                I.getName(), nullptr, I.getThreadLocalMode(),
                                I.getType()->getAddressSpace());
      VMap[&I] = GV;
      // We do not copy attributes (mainly because copying between different
      // kinds of globals is forbidden), but this is generally not required for
      // correctness.
      continue;
    }
    auto *GA = GlobalAlias::create(I.getValueType(),
                                   I.getType()->getPointerAddressSpace(),
                                   I.getLinkage(), I.getName(), New.get());
    GA->copyAttributesFrom(&I);
    VMap[&I] = GA;
  }

  for (const GlobalIFunc &I : M.ifuncs()) {
    // Defer setting the resolver function until after functions are cloned.
    auto *GI =
        GlobalIFunc::create(I.getValueType(), I.getAddressSpace(),
                            I.getLinkage(), I.getName(), nullptr, New.get());
    GI->copyAttributesFrom(&I);
    VMap[&I] = GI;
  }

  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (const GlobalVariable &G : M.globals()) {
    GlobalVariable *GV = cast<GlobalVariable>(VMap[&G]);

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    G.getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(MD.first, *MapMetadata(MD.second, VMap));

    if (G.isDeclaration())
      continue;

    if (!ShouldCloneDefinition(&G)) {
      // Skip after setting the correct linkage for an external reference.
      GV->setLinkage(GlobalValue::ExternalLinkage);
      continue;
    }
    if (G.hasInitializer())
      GV->setInitializer(MapValue(G.getInitializer(), VMap));

    copyComdat(GV, &G);
  }

  // Similarly, copy over function bodies now...
  //
  for (const Function &I : M) {
    Function *F = cast<Function>(VMap[&I]);

    if (I.isDeclaration()) {
      // Copy over metadata for declarations since we're not doing it below in
      // CloneFunctionInto().
      SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
      I.getAllMetadata(MDs);
      for (auto MD : MDs)
        F->addMetadata(MD.first, *MapMetadata(MD.second, VMap));
      continue;
    }

    if (!ShouldCloneDefinition(&I)) {
      // Skip after setting the correct linkage for an external reference.
      F->setLinkage(GlobalValue::ExternalLinkage);
      // Personality function is not valid on a declaration.
      F->setPersonalityFn(nullptr);
      continue;
    }

    Function::arg_iterator DestI = F->arg_begin();
    for (const Argument &J : I.args()) {
      DestI->setName(J.getName());
      VMap[&J] = &*DestI++;
    }

    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(F, &I, VMap, CloneFunctionChangeType::ClonedModule,
                      Returns);

    if (I.hasPersonalityFn())
      F->setPersonalityFn(MapValue(I.getPersonalityFn(), VMap));

    copyComdat(F, &I);
  }

  // And aliases
  for (const GlobalAlias &I : M.aliases()) {
    // We already dealt with undefined aliases above.
    if (!ShouldCloneDefinition(&I))
      continue;
    GlobalAlias *GA = cast<GlobalAlias>(VMap[&I]);
    if (const Constant *C = I.getAliasee())
      GA->setAliasee(MapValue(C, VMap));
  }

  for (const GlobalIFunc &I : M.ifuncs()) {
    GlobalIFunc *GI = cast<GlobalIFunc>(VMap[&I]);
    if (const Constant *Resolver = I.getResolver())
      GI->setResolver(MapValue(Resolver, VMap));
  }

  // And named metadata....
  for (const NamedMDNode &NMD : M.named_metadata()) {
    NamedMDNode *NewNMD = New->getOrInsertNamedMetadata(NMD.getName());
    for (const MDNode *N : NMD.operands())
      NewNMD->addOperand(MapMetadata(N, VMap));
  }

  //Now add the instrumentation function passing in the added ptr kernel argument as the bufferPtr
  uint32_t LocationCounter = 0;
  for (Function::iterator BB = NF->begin(); BB != NF->end(); BB++) {
    for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
      if (LoadInst* LI = dyn_cast<LoadInst>(I)) {
          InjectInstrumentationFunction<LoadInst>(I, *NF, *New.get(), LocationCounter, bufferPtr, false);
      }
      else if(StoreInst* SI = dyn_cast<StoreInst>(I)){
          InjectInstrumentationFunction<StoreInst>(I, *NF, *New.get(), LocationCounter, bufferPtr, false);
      }
    }
  } 
  return New;
}

void AddArg(const Function &F, const llvm::Module &M){
	ValueToValueMapTy VMap;
	std::unique_ptr<Module> ClonedModule = CloneModuleAndAddArg(M, VMap, F,  [](const GlobalValue *GV) { return true; });
	std::string moduleName = F.getName().str() + ".bc";
   	std::error_code EC;
   	llvm::raw_fd_ostream OS(moduleName, EC, llvm::sys::fs::OF_None);
   	WriteBitcodeToFile(*ClonedModule.get(), OS);
   	OS.flush();	
	return;	
}




bool AMDGCNMemTrace::runOnModule(Module &M) {
  bool ModifiedCodeGen = false;
  auto &CTX = M.getContext();
  uint32_t LocationCounter = 0;
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
	    AddArg(F, M);
      for (Function::iterator BB = F.begin(); BB != F.end(); BB++) {
        for (BasicBlock::iterator I = BB->begin(); I != BB->end(); I++) {
          if (LoadInst* LI = dyn_cast<LoadInst>(I)) {      
	      PointerType *VoidPtrType = PointerType::getUnqual(CTX);
	      Constant *NullPtrVal = ConstantPointerNull::get(VoidPtrType);
	      InjectInstrumentationFunction<LoadInst>(I, F, M, LocationCounter, NullPtrVal, true);
              ModifiedCodeGen = true;                                                                     
          }
	  else if(StoreInst* SI = dyn_cast<StoreInst>(I)){
		  PointerType *VoidPtrType = PointerType::getUnqual(CTX);
		  Constant *NullPtrVal = ConstantPointerNull::get(VoidPtrType);
		  InjectInstrumentationFunction<StoreInst>(I, F, M, LocationCounter, NullPtrVal, true);
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
