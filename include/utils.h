#ifndef UTILS_H
#define UTILS_H

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/StringExtras.h"
#include <set>

using namespace llvm;
using namespace std;

namespace instrumentation::utils {

inline std::string getenv(const char *name) {
  const char *cstr = std::getenv(name);
  if (!cstr)
    return "";
  std::string result(cstr);
  return result;
}
}

inline static void drop_front(StringRef& str, size_t n = 1) {
  str = str.drop_front(n);
}

static int eatNumber(StringRef& s) {
  size_t const savedSize = s.size();
  int n = 0;
  while (!s.empty() && isDigit(s.front())) {
    n = n*10 + s.front() - '0';
    drop_front(s);
  }
  return s.size() < savedSize ? n : -1;
}

static StringRef eatLengthPrefixedName(StringRef& mangledName) {
  int const Len = eatNumber(mangledName);
  if (Len <= 0 || static_cast<size_t>(Len) > mangledName.size())
    return StringRef();
  StringRef Res = mangledName.substr(0, Len);
  drop_front(mangledName, Len);
  return Res;
}

template <size_t N>
static bool eatTerm(StringRef& mangledName, const char (&str)[N]) {
  if (mangledName.starts_with(StringRef(str, N - 1))) {
    drop_front(mangledName, N-1);
    return true;
  }
  return false;
}

static StringRef getUnmangledName(StringRef mangledName) {
  StringRef S = mangledName;
  if (eatTerm(S, "_Z"))
    return eatLengthPrefixedName(S);
  return StringRef();
}

Type *parseBasicTypeName(StringRef &TypeName, LLVMContext &Ctx) {
  if (TypeName.consume_front("void"))
    return Type::getVoidTy(Ctx);
  else if (TypeName.consume_front("bool"))
    return Type::getIntNTy(Ctx, 1);
  else if (TypeName.consume_front("char") ||
           TypeName.consume_front("unsigned char") ||
           TypeName.consume_front("uchar"))
    return Type::getInt8Ty(Ctx);
  else if (TypeName.consume_front("short") ||
           TypeName.consume_front("unsigned short") ||
           TypeName.consume_front("ushort"))
    return Type::getInt16Ty(Ctx);
  else if (TypeName.consume_front("int") ||
           TypeName.consume_front("unsigned int") ||
           TypeName.consume_front("uint"))
    return Type::getInt32Ty(Ctx);
  else if (TypeName.consume_front("long") ||
           TypeName.consume_front("unsigned long") ||
           TypeName.consume_front("ulong"))
    return Type::getInt64Ty(Ctx);
  else if (TypeName.consume_front("half"))
    return Type::getHalfTy(Ctx);
  else if (TypeName.consume_front("float"))
    return Type::getFloatTy(Ctx);
  else if (TypeName.consume_front("double"))
    return Type::getDoubleTy(Ctx);

  // Unable to recognize  type name
  return nullptr;
}

bool loadInstrumentationFile(const std::string &fileName, LLVMContext &context,
                    std::unique_ptr<llvm::Module> &modules,
                    std::string &errorMsg) {

  ErrorOr<std::unique_ptr<MemoryBuffer>> bufferErr =
      MemoryBuffer::getFileOrSTDIN(fileName);
  std::error_code ec = bufferErr.getError();
  if (ec) {
    printf("Loading file %s failed: %s", fileName.c_str(),
               ec.message().c_str());
	  exit(1);
  }

  MemoryBufferRef Buffer = bufferErr.get()->getMemBufferRef();
  file_magic magic = identify_magic(Buffer.getBuffer());

  if (magic == file_magic::bitcode) {
    SMDiagnostic Err;
    std::unique_ptr<llvm::Module> module(parseIR(Buffer, Err, context));
    if (!module) {
      printf("Loading file %s failed: %s", fileName.c_str(),
                 Err.getMessage().str().c_str());
	  exit(1);
    }
    modules = std::move(module);
    return true;
  }

   errorMsg = "Loading file " + fileName +
              " Object file as input is currently not supported";
   return false;
}

static void
GetAllUndefinedSymbols(llvm::Module *M, std::set<std::string> &UndefinedSymbols) {
  static const std::string llvmIntrinsicPrefix="llvm.";
  std::set<std::string> DefinedSymbols;
  UndefinedSymbols.clear();

  for (auto const &Function : *M) {
    if (Function.hasName()) {
      if (Function.isDeclaration())
        UndefinedSymbols.insert(Function.getName().str());
      else if (!Function.hasLocalLinkage()) {
        assert(!Function.hasDLLImportStorageClass() &&
               "Found dllimported non-external symbol!");
        DefinedSymbols.insert(Function.getName().str());
      }
    }
  }

  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    if (I->hasName()) {
      if (I->isDeclaration())
        UndefinedSymbols.insert(I->getName().str());
      else if (!I->hasLocalLinkage()) {
        assert(!I->hasDLLImportStorageClass() && "Found dllimported non-external symbol!");
        DefinedSymbols.insert(I->getName().str());
      }
    }

  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I)
    if (I->hasName())
      DefinedSymbols.insert(I->getName().str());

  // Prune out any defined symbols from the undefined symbols set
  // and other symbols we don't want to treat as an undefined symbol
  std::vector<std::string> SymbolsToRemove;
  for (std::set<std::string>::iterator I = UndefinedSymbols.begin();
       I != UndefinedSymbols.end(); ++I )
  {
    if (DefinedSymbols.find(*I) != DefinedSymbols.end()) {
      SymbolsToRemove.push_back(*I);
      continue;
    }

    // Strip out llvm intrinsics
    if ( (I->size() >= llvmIntrinsicPrefix.size() ) &&
       (I->compare(0, llvmIntrinsicPrefix.size(), llvmIntrinsicPrefix) == 0) )
    {
      SymbolsToRemove.push_back(*I);
      continue;
    }
  }

  // Now remove the symbols from undefined set.
  for (auto const &symbol : SymbolsToRemove)
    UndefinedSymbols.erase(symbol);
}

static bool linkTwoModules(llvm::Module *Dest,
                           std::unique_ptr<llvm::Module> Src,
                           std::string &errorMsg) {
  // Get the potential error message (Src is moved and won't be available later)
  errorMsg = "Linking module " + Src->getModuleIdentifier() + " failed";
  auto linkResult = Linker::linkModules(*Dest, std::move(Src));

  return !linkResult;
} 

void linkModules(llvm::Module &mainModule, llvm::Module &instrumentationModule, std::string &errorMsg){

  return;
}

std::unique_ptr<llvm::Module>
linkModules(std::vector<std::unique_ptr<llvm::Module>> &modules,
                  llvm::StringRef entryFunction, std::string &errorMsg) {
  assert(!modules.empty() && "modules list should not be empty");

  if (entryFunction.empty()) {
    // If no entry function is provided, link all modules together into one
    std::unique_ptr<llvm::Module> composite = std::move(modules.back());
    modules.pop_back();

    // Just link all modules together
    for (auto &module : modules) {
      if (linkTwoModules(composite.get(), std::move(module), errorMsg))
        continue;

      // Linking failed
      errorMsg = "Linking archive module with composite failed:" + errorMsg;
      return nullptr;
    }

    // clean up every module as we already linked in every module
    modules.clear();
    return composite;
  }

  // Starting from the module containing the entry function, resolve unresolved
  // dependencies recursively


  // search for the module containing the entry function
  std::unique_ptr<llvm::Module> composite;
  for (auto &module : modules) {
    if (!module || !module->getNamedValue(entryFunction))
      continue;
    if (composite) {
      errorMsg =
          "Function " + entryFunction.str() +
          " defined in different modules (" + module->getModuleIdentifier() +
          " already defined in: " + composite->getModuleIdentifier() + ")";
      return nullptr;
    }
    composite = std::move(module);
  }

  // fail if not found
  if (!composite) {
    errorMsg =
        "Entry function '" + entryFunction.str() + "' not found in module.";
    return nullptr;
  }

  auto containsUsedSymbols = [](const llvm::Module *module) {
    GlobalValue *GV =
        dyn_cast_or_null<GlobalValue>(module->getNamedValue("llvm.used"));
    if (!GV)
      return false;
    return true;
  };

  for (auto &module : modules) {
    if (!module || !containsUsedSymbols(module.get()))
      continue;
    if (!linkTwoModules(composite.get(), std::move(module), errorMsg)) {
      // Linking failed
      errorMsg = "Linking module containing '__attribute__((used))'"
                 " symbols with composite failed:" +
                 errorMsg;
      return nullptr;
    }
    module = nullptr;
  }

  bool symbolsLinked = true;
  while (symbolsLinked) {
    symbolsLinked = false;
    std::set<std::string> undefinedSymbols;
    GetAllUndefinedSymbols(composite.get(), undefinedSymbols);
    auto hasRequiredDefinition = [&undefinedSymbols](
                                     const llvm::Module *module) {
      for (auto symbol : undefinedSymbols) {
        GlobalValue *GV =
            dyn_cast_or_null<GlobalValue>(module->getNamedValue(symbol));
        if (GV && !GV->isDeclaration()) {
          return true;
        }
      }
      return false;
    };

    // Stop in nothing is undefined
    if (undefinedSymbols.empty())
      break;

    for (auto &module : modules) {
      if (!module)
        continue;

      if (!hasRequiredDefinition(module.get()))
        continue;

      if (!linkTwoModules(composite.get(), std::move(module), errorMsg)) {
        // Linking failed
        errorMsg = "Linking archive module with composite failed:" + errorMsg;
        return nullptr;
      }
      module = nullptr;
      symbolsLinked = true;
    }
  }

  // Condense the module array
  std::vector<std::unique_ptr<llvm::Module>> LeftoverModules;
  for (auto &module : modules) {
    if (module)
      LeftoverModules.emplace_back(std::move(module));
  }

  modules.swap(LeftoverModules);
  return composite;
}

#endif
