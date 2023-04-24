"""
This type stub file was generated by pyright.
"""

"""psCharStrings.py -- module implementing various kinds of CharStrings:
CFF dictionary data and Type1/Type2 CharStrings.
"""
log = ...
def read_operator(self, b0, data, index): # -> tuple[None, Unknown] | tuple[Unknown, Unknown]:
    ...

def read_byte(self, b0, data, index): # -> tuple[Unknown, Unknown]:
    ...

def read_smallInt1(self, b0, data, index): # -> tuple[Unknown, Unknown]:
    ...

def read_smallInt2(self, b0, data, index): # -> tuple[Unknown, Unknown]:
    ...

def read_shortInt(self, b0, data, index): # -> tuple[Any, Unknown]:
    ...

def read_longInt(self, b0, data, index): # -> tuple[Any, Unknown]:
    ...

def read_fixed1616(self, b0, data, index): # -> tuple[Any, Unknown]:
    ...

def read_reserved(self, b0, data, index): # -> tuple[_NotImplementedType, Unknown]:
    ...

def read_realNumber(self, b0, data, index): # -> tuple[float, Unknown]:
    ...

t1OperandEncoding = ...
t2OperandEncoding = ...
cffDictOperandEncoding = ...
realNibbles = ...
realNibblesDict = ...
maxOpStack = ...
def buildOperatorDict(operatorList): # -> tuple[dict[Unknown, Unknown], dict[Unknown, Unknown]]:
    ...

t2Operators = ...
def getIntEncoder(format): # -> (value: Unknown, fourByteOp: bytes | None = fourByteOp, bytechr: (n: Unknown) -> bytes = bytechr, pack: (__fmt: str | bytes, *v: Any) -> bytes = struct.pack, unpack: (__format: str | bytes, __buffer: ReadableBuffer, /) -> tuple[Any, ...] = struct.unpack) -> bytes:
    ...

encodeIntCFF = ...
encodeIntT1 = ...
encodeIntT2 = ...
def encodeFixed(f, pack=...): # -> bytes:
    """For T2 only"""
    ...

realZeroBytes = ...
def encodeFloat(f): # -> bytes:
    ...

class CharStringCompileError(Exception):
    ...


class SimpleT2Decompiler:
    def __init__(self, localSubrs, globalSubrs, private=..., blender=...) -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def execute(self, charString): # -> None:
        ...
    
    def pop(self):
        ...
    
    def popall(self): # -> list[Unknown]:
        ...
    
    def push(self, value): # -> None:
        ...
    
    def op_return(self, index): # -> None:
        ...
    
    def op_endchar(self, index): # -> None:
        ...
    
    def op_ignore(self, index): # -> None:
        ...
    
    def op_callsubr(self, index): # -> None:
        ...
    
    def op_callgsubr(self, index): # -> None:
        ...
    
    def op_hstem(self, index): # -> None:
        ...
    
    def op_vstem(self, index): # -> None:
        ...
    
    def op_hstemhm(self, index): # -> None:
        ...
    
    def op_vstemhm(self, index): # -> None:
        ...
    
    def op_hintmask(self, index): # -> tuple[Unknown, Unknown]:
        ...
    
    op_cntrmask = ...
    def countHints(self): # -> None:
        ...
    
    def op_and(self, index):
        ...
    
    def op_or(self, index):
        ...
    
    def op_not(self, index):
        ...
    
    def op_store(self, index):
        ...
    
    def op_abs(self, index):
        ...
    
    def op_add(self, index):
        ...
    
    def op_sub(self, index):
        ...
    
    def op_div(self, index):
        ...
    
    def op_load(self, index):
        ...
    
    def op_neg(self, index):
        ...
    
    def op_eq(self, index):
        ...
    
    def op_drop(self, index):
        ...
    
    def op_put(self, index):
        ...
    
    def op_get(self, index):
        ...
    
    def op_ifelse(self, index):
        ...
    
    def op_random(self, index):
        ...
    
    def op_mul(self, index):
        ...
    
    def op_sqrt(self, index):
        ...
    
    def op_dup(self, index):
        ...
    
    def op_exch(self, index):
        ...
    
    def op_index(self, index):
        ...
    
    def op_roll(self, index):
        ...
    
    def op_blend(self, index): # -> None:
        ...
    
    def op_vsindex(self, index): # -> None:
        ...
    


t1Operators = ...
class T2WidthExtractor(SimpleT2Decompiler):
    def __init__(self, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private=..., blender=...) -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def popallWidth(self, evenOdd=...): # -> list[Unknown]:
        ...
    
    def countHints(self): # -> None:
        ...
    
    def op_rmoveto(self, index): # -> None:
        ...
    
    def op_hmoveto(self, index): # -> None:
        ...
    
    def op_vmoveto(self, index): # -> None:
        ...
    
    def op_endchar(self, index): # -> None:
        ...
    


class T2OutlineExtractor(T2WidthExtractor):
    def __init__(self, pen, localSubrs, globalSubrs, nominalWidthX, defaultWidthX, private=..., blender=...) -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def execute(self, charString): # -> None:
        ...
    
    def rMoveTo(self, point): # -> None:
        ...
    
    def rLineTo(self, point): # -> None:
        ...
    
    def rCurveTo(self, pt1, pt2, pt3): # -> None:
        ...
    
    def closePath(self): # -> None:
        ...
    
    def endPath(self): # -> None:
        ...
    
    def op_rmoveto(self, index): # -> None:
        ...
    
    def op_hmoveto(self, index): # -> None:
        ...
    
    def op_vmoveto(self, index): # -> None:
        ...
    
    def op_endchar(self, index): # -> None:
        ...
    
    def op_rlineto(self, index): # -> None:
        ...
    
    def op_hlineto(self, index): # -> None:
        ...
    
    def op_vlineto(self, index): # -> None:
        ...
    
    def op_rrcurveto(self, index): # -> None:
        """{dxa dya dxb dyb dxc dyc}+ rrcurveto"""
        ...
    
    def op_rcurveline(self, index): # -> None:
        """{dxa dya dxb dyb dxc dyc}+ dxd dyd rcurveline"""
        ...
    
    def op_rlinecurve(self, index): # -> None:
        """{dxa dya}+ dxb dyb dxc dyc dxd dyd rlinecurve"""
        ...
    
    def op_vvcurveto(self, index): # -> None:
        "dx1? {dya dxb dyb dyc}+ vvcurveto"
        ...
    
    def op_hhcurveto(self, index): # -> None:
        """dy1? {dxa dxb dyb dxc}+ hhcurveto"""
        ...
    
    def op_vhcurveto(self, index): # -> None:
        """dy1 dx2 dy2 dx3 {dxa dxb dyb dyc dyd dxe dye dxf}* dyf? vhcurveto (30)
        {dya dxb dyb dxc dxd dxe dye dyf}+ dxf? vhcurveto
        """
        ...
    
    def op_hvcurveto(self, index): # -> None:
        """dx1 dx2 dy2 dy3 {dya dxb dyb dxc dxd dxe dye dyf}* dxf?
        {dxa dxb dyb dyc dyd dxe dye dxf}+ dyf?
        """
        ...
    
    def op_hflex(self, index): # -> None:
        ...
    
    def op_flex(self, index): # -> None:
        ...
    
    def op_hflex1(self, index): # -> None:
        ...
    
    def op_flex1(self, index): # -> None:
        ...
    
    def op_and(self, index):
        ...
    
    def op_or(self, index):
        ...
    
    def op_not(self, index):
        ...
    
    def op_store(self, index):
        ...
    
    def op_abs(self, index):
        ...
    
    def op_add(self, index):
        ...
    
    def op_sub(self, index):
        ...
    
    def op_div(self, index): # -> None:
        ...
    
    def op_load(self, index):
        ...
    
    def op_neg(self, index):
        ...
    
    def op_eq(self, index):
        ...
    
    def op_drop(self, index):
        ...
    
    def op_put(self, index):
        ...
    
    def op_get(self, index):
        ...
    
    def op_ifelse(self, index):
        ...
    
    def op_random(self, index):
        ...
    
    def op_mul(self, index):
        ...
    
    def op_sqrt(self, index):
        ...
    
    def op_dup(self, index):
        ...
    
    def op_exch(self, index):
        ...
    
    def op_index(self, index):
        ...
    
    def op_roll(self, index):
        ...
    
    def alternatingLineto(self, isHorizontal): # -> None:
        ...
    
    def vcurveto(self, args): # -> list[Unknown]:
        ...
    
    def hcurveto(self, args): # -> list[Unknown]:
        ...
    


class T1OutlineExtractor(T2OutlineExtractor):
    def __init__(self, pen, subrs) -> None:
        ...
    
    def reset(self): # -> None:
        ...
    
    def endPath(self): # -> None:
        ...
    
    def popallWidth(self, evenOdd=...): # -> list[Unknown]:
        ...
    
    def exch(self): # -> None:
        ...
    
    def op_rmoveto(self, index): # -> None:
        ...
    
    def op_hmoveto(self, index): # -> None:
        ...
    
    def op_vmoveto(self, index): # -> None:
        ...
    
    def op_closepath(self, index): # -> None:
        ...
    
    def op_setcurrentpoint(self, index): # -> None:
        ...
    
    def op_endchar(self, index): # -> None:
        ...
    
    def op_hsbw(self, index): # -> None:
        ...
    
    def op_sbw(self, index): # -> None:
        ...
    
    def op_callsubr(self, index): # -> None:
        ...
    
    def op_callothersubr(self, index): # -> None:
        ...
    
    def op_pop(self, index): # -> None:
        ...
    
    def doFlex(self): # -> None:
        ...
    
    def op_dotsection(self, index): # -> None:
        ...
    
    def op_hstem3(self, index): # -> None:
        ...
    
    def op_seac(self, index): # -> None:
        "asb adx ady bchar achar seac"
        ...
    
    def op_vstem3(self, index): # -> None:
        ...
    


class T2CharString:
    operandEncoding = ...
    decompilerClass = SimpleT2Decompiler
    outlineExtractor = T2OutlineExtractor
    def __init__(self, bytecode=..., program=..., private=..., globalSubrs=...) -> None:
        ...
    
    def getNumRegions(self, vsindex=...):
        ...
    
    def __repr__(self): # -> str:
        ...
    
    def getIntEncoder(self): # -> (value: Unknown, fourByteOp: bytes | None = fourByteOp, bytechr: (n: Unknown) -> bytes = bytechr, pack: (__fmt: str | bytes, *v: Any) -> bytes = struct.pack, unpack: (__format: str | bytes, __buffer: ReadableBuffer, /) -> tuple[Any, ...] = struct.unpack) -> bytes:
        ...
    
    def getFixedEncoder(self): # -> (f: Unknown, pack: (__fmt: str | bytes, *v: Any) -> bytes = struct.pack) -> bytes:
        ...
    
    def decompile(self): # -> None:
        ...
    
    def draw(self, pen, blender=...): # -> None:
        ...
    
    def calcBounds(self, glyphSet): # -> tuple[Unknown, Unknown, Unknown, Unknown]:
        ...
    
    def compile(self, isCFF2=...): # -> None:
        ...
    
    def needsDecompilation(self): # -> bool:
        ...
    
    def setProgram(self, program): # -> None:
        ...
    
    def setBytecode(self, bytecode): # -> None:
        ...
    
    def getToken(self, index, len=..., byteord=..., isinstance=...): # -> tuple[None, Literal[0], Literal[0]] | tuple[Unknown, bool, Unknown]:
        ...
    
    def getBytes(self, index, nBytes): # -> tuple[Unknown, Unknown]:
        ...
    
    def handle_operator(self, operator):
        ...
    
    def toXML(self, xmlWriter, ttFont=...): # -> None:
        ...
    
    def fromXML(self, name, attrs, content): # -> None:
        ...
    


class T1CharString(T2CharString):
    operandEncoding = ...
    def __init__(self, bytecode=..., program=..., subrs=...) -> None:
        ...
    
    def getIntEncoder(self): # -> (value: Unknown, fourByteOp: bytes | None = fourByteOp, bytechr: (n: Unknown) -> bytes = bytechr, pack: (__fmt: str | bytes, *v: Any) -> bytes = struct.pack, unpack: (__format: str | bytes, __buffer: ReadableBuffer, /) -> tuple[Any, ...] = struct.unpack) -> bytes:
        ...
    
    def getFixedEncoder(self): # -> None:
        ...
    
    def decompile(self): # -> None:
        ...
    
    def draw(self, pen): # -> None:
        ...
    


class DictDecompiler:
    operandEncoding = ...
    def __init__(self, strings, parent=...) -> None:
        ...
    
    def getDict(self): # -> dict[Unknown, Unknown]:
        ...
    
    def decompile(self, data): # -> None:
        ...
    
    def pop(self):
        ...
    
    def popall(self): # -> list[Unknown]:
        ...
    
    def handle_operator(self, operator): # -> None:
        ...
    
    def arg_number(self, name):
        ...
    
    def arg_blend_number(self, name):
        ...
    
    def arg_SID(self, name):
        ...
    
    def arg_array(self, name): # -> list[Unknown]:
        ...
    
    def arg_blendList(self, name):
        """
        There may be non-blend args at the top of the stack. We first calculate
        where the blend args start in the stack. These are the last
        numMasters*numBlends) +1 args.
        The blend args starts with numMasters relative coordinate values, the  BlueValues in the list from the default master font. This is followed by
        numBlends list of values. Each of  value in one of these lists is the
        Variable Font delta for the matching region.

        We re-arrange this to be a list of numMaster entries. Each entry starts with the corresponding default font relative value, and is followed by
        the delta values. We then convert the default values, the first item in each entry, to an absolute value.
        """
        ...
    
    def arg_delta(self, name): # -> list[Unknown]:
        ...
    


def calcSubrBias(subrs): # -> Literal[107, 1131, 32768]:
    ...

