using namespace llvm;
using namespace std;

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