#ifndef INJECT_AMDGCN_FUNC_H
#define INJECT_AMDGCN_FUNC_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace {

struct InjectAMDGCNFunc : public PassInfoMixin<InjectAMDGCNFunc> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    bool Changed = runOnModule(M);

    return (Changed ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all());
  }
  bool runOnModule(llvm::Module &M);
  // isRequired being set to true keeps this pass from being skipped
  // if the function has the optnone LLVM attribute
  static bool isRequired() { return true; }
};

} // end anonymous namespace

#endif
