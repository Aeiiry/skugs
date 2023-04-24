"""
This type stub file was generated by pyright.
"""

from .otTables import CompositeMode as _CompositeMode, ExtendMode as _ExtendMode
from typing import Optional

log = ...
istuple = ...
def buildConverters(tableSpec, tableNamespace): # -> tuple[list[Unknown], dict[Unknown, Unknown]]:
    """Given a table spec from otData.py, build a converter object for each
    field of the table. This is called for each table in otData.py, and
    the results are assigned to the corresponding class in otTables.py."""
    ...

class _MissingItem(tuple):
    __slots__ = ...


class _LazyList(UserList):
    def __getslice__(self, i, j): # -> list[Unknown]:
        ...
    
    def __getitem__(self, k): # -> list[Unknown]:
        ...
    
    def __add__(self, other): # -> _NotImplementedType | list[Unknown]:
        ...
    
    def __radd__(self, other): # -> _NotImplementedType | list[Unknown]:
        ...
    


class BaseConverter:
    """Base class for converter objects. Apart from the constructor, this
    is an abstract class."""
    def __init__(self, name, repeat, aux, tableClass=..., *, description=...) -> None:
        ...
    
    def readArray(self, reader, font, tableDict, count): # -> list[Unknown] | _LazyList:
        """Read an array of values from the reader."""
        ...
    
    def getRecordSize(self, reader): # -> _NotImplementedType:
        ...
    
    def read(self, reader, font, tableDict):
        """Read a value from the reader."""
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...):
        """Write a value to the writer."""
        ...
    
    def xmlRead(self, attrs, content, font):
        """Read a value from XML."""
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        """Write a value to XML."""
        ...
    
    varIndexBasePlusOffsetRE = ...
    def getVarIndexOffset(self) -> Optional[int]:
        """If description has `VarIndexBase + {offset}`, return the offset else None."""
        ...
    


class SimpleValue(BaseConverter):
    @staticmethod
    def toString(value):
        ...
    
    @staticmethod
    def fromString(value):
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font):
        ...
    


class OptionalValue(SimpleValue):
    DEFAULT = ...
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> None:
        ...
    


class IntValue(SimpleValue):
    @staticmethod
    def fromString(value): # -> int:
        ...
    


class Long(IntValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def readArray(self, reader, font, tableDict, count):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    


class ULong(IntValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def readArray(self, reader, font, tableDict, count):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    


class Flags32(ULong):
    @staticmethod
    def toString(value):
        ...
    


class VarIndex(OptionalValue, ULong):
    DEFAULT = ...


class Short(IntValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def readArray(self, reader, font, tableDict, count):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    


class UShort(IntValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def readArray(self, reader, font, tableDict, count):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    


class Int8(IntValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def readArray(self, reader, font, tableDict, count):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    


class UInt8(IntValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def readArray(self, reader, font, tableDict, count):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    


class UInt24(IntValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class ComputedInt(IntValue):
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class ComputedUInt8(ComputedInt, UInt8):
    ...


class ComputedUShort(ComputedInt, UShort):
    ...


class ComputedULong(ComputedInt, ULong):
    ...


class Tag(SimpleValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class GlyphID(SimpleValue):
    staticSize = ...
    typecode = ...
    def readArray(self, reader, font, tableDict, count):
        ...
    
    def read(self, reader, font, tableDict):
        ...
    
    def writeArray(self, writer, font, tableDict, values): # -> None:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class GlyphID32(GlyphID):
    staticSize = ...
    typecode = ...


class NameID(UShort):
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class STATFlags(UShort):
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class FloatValue(SimpleValue):
    @staticmethod
    def fromString(value): # -> float:
        ...
    


class DeciPoints(FloatValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class BaseFixedValue(FloatValue):
    staticSize = ...
    precisionBits = ...
    readerMethod = ...
    writerMethod = ...
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    @classmethod
    def fromInt(cls, value):
        ...
    
    @classmethod
    def toInt(cls, value): # -> int:
        ...
    
    @classmethod
    def fromString(cls, value): # -> float:
        ...
    
    @classmethod
    def toString(cls, value): # -> str:
        ...
    


class Fixed(BaseFixedValue):
    staticSize = ...
    precisionBits = ...
    readerMethod = ...
    writerMethod = ...


class F2Dot14(BaseFixedValue):
    staticSize = ...
    precisionBits = ...
    readerMethod = ...
    writerMethod = ...


class Angle(F2Dot14):
    bias = ...
    factor = ...
    @classmethod
    def fromInt(cls, value):
        ...
    
    @classmethod
    def toInt(cls, value): # -> int:
        ...
    
    @classmethod
    def fromString(cls, value): # -> float:
        ...
    
    @classmethod
    def toString(cls, value): # -> str:
        ...
    


class BiasedAngle(Angle):
    bias = ...


class Version(SimpleValue):
    staticSize = ...
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    @staticmethod
    def fromString(value): # -> int | float:
        ...
    
    @staticmethod
    def toString(value):
        ...
    
    @staticmethod
    def fromFloat(v): # -> int:
        ...
    


class Char64(SimpleValue):
    """An ASCII string with up to 64 characters.

    Unused character positions are filled with 0x00 bytes.
    Used in Apple AAT fonts in the `gcid` table.
    """
    staticSize = ...
    def read(self, reader, font, tableDict): # -> str:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class Struct(BaseConverter):
    def getRecordSize(self, reader): # -> None:
        ...
    
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class StructWithLength(Struct):
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class Table(Struct):
    staticSize = ...
    def readOffset(self, reader):
        ...
    
    def writeNullOffset(self, writer): # -> None:
        ...
    
    def read(self, reader, font, tableDict): # -> None:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class LTable(Table):
    staticSize = ...
    def readOffset(self, reader):
        ...
    
    def writeNullOffset(self, writer): # -> None:
        ...
    


class Table24(Table):
    staticSize = ...
    def readOffset(self, reader):
        ...
    
    def writeNullOffset(self, writer): # -> None:
        ...
    


class SubStruct(Struct):
    def getConverter(self, tableType, lookupType): # -> SubStruct:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class SubTable(Table):
    def getConverter(self, tableType, lookupType): # -> SubTable:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class ExtSubTable(LTable, SubTable):
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class FeatureParams(Table):
    def getConverter(self, featureTag): # -> FeatureParams:
        ...
    


class ValueFormat(IntValue):
    staticSize = ...
    def __init__(self, name, repeat, aux, tableClass=..., *, description=...) -> None:
        ...
    
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, format, repeatIndex=...): # -> None:
        ...
    


class ValueRecord(ValueFormat):
    def getRecordSize(self, reader): # -> int:
        ...
    
    def read(self, reader, font, tableDict):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> ValueRecord:
        ...
    


class AATLookup(BaseConverter):
    BIN_SEARCH_HEADER_SIZE = ...
    def __init__(self, name, repeat, aux, tableClass, *, description=...) -> None:
        ...
    
    def read(self, reader, font, tableDict): # -> dict[Unknown, Unknown]:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    @staticmethod
    def writeBinSearchHeader(writer, numUnits, unitSize): # -> None:
        ...
    
    def buildFormat0(self, writer, font, values): # -> tuple[Unknown | int, Literal[0], () -> None] | None:
        ...
    
    def writeFormat0(self, writer, font, values): # -> None:
        ...
    
    def buildFormat2(self, writer, font, values): # -> tuple[Unknown | int, Literal[2], () -> None]:
        ...
    
    def writeFormat2(self, writer, font, segments): # -> None:
        ...
    
    def buildFormat6(self, writer, font, values): # -> tuple[Unknown | int, Literal[6], () -> None]:
        ...
    
    def writeFormat6(self, writer, font, values): # -> None:
        ...
    
    def buildFormat8(self, writer, font, values): # -> tuple[Unknown | int, Literal[8], () -> None] | None:
        ...
    
    def writeFormat8(self, writer, font, values): # -> None:
        ...
    
    def readFormat0(self, reader, font): # -> dict[Unknown, Unknown]:
        ...
    
    def readFormat2(self, reader, font): # -> dict[Unknown, Unknown]:
        ...
    
    def readFormat4(self, reader, font): # -> dict[Unknown, Unknown]:
        ...
    
    def readFormat6(self, reader, font): # -> dict[Unknown, Unknown]:
        ...
    
    def readFormat8(self, reader, font): # -> dict[Unknown, Unknown]:
        ...
    
    def xmlRead(self, attrs, content, font): # -> dict[Unknown, Unknown]:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class AATLookupWithDataOffset(BaseConverter):
    def read(self, reader, font, tableDict): # -> dict[Unknown, Unknown]:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> dict[Unknown, Unknown]:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class MorxSubtableConverter(BaseConverter):
    _PROCESSING_ORDERS = ...
    _PROCESSING_ORDERS_REVERSED = ...
    def __init__(self, name, repeat, aux, tableClass=..., *, description=...) -> None:
        ...
    
    def read(self, reader, font, tableDict):
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font):
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class STXHeader(BaseConverter):
    def __init__(self, name, repeat, aux, tableClass, *, description=...) -> None:
        ...
    
    def read(self, reader, font, tableDict): # -> AATStateTable:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> AATStateTable:
        ...
    


class CIDGlyphMap(BaseConverter):
    def read(self, reader, font, tableDict): # -> dict[Unknown, Unknown]:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> dict[Unknown, Unknown]:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class GlyphCIDMap(BaseConverter):
    def read(self, reader, font, tableDict): # -> dict[Unknown, Unknown]:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> dict[Unknown, Unknown]:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class DeltaValue(BaseConverter):
    def read(self, reader, font, tableDict): # -> list[Unknown]:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> Any:
        ...
    


class VarIdxMapValue(BaseConverter):
    def read(self, reader, font, tableDict): # -> list[Unknown]:
        ...
    
    def write(self, writer, font, tableDict, value, repeatIndex=...): # -> None:
        ...
    


class VarDataValue(BaseConverter):
    def read(self, reader, font, tableDict): # -> list[Unknown]:
        ...
    
    def write(self, writer, font, tableDict, values, repeatIndex=...): # -> None:
        ...
    
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    
    def xmlRead(self, attrs, content, font): # -> Any:
        ...
    


class LookupFlag(UShort):
    def xmlWrite(self, xmlWriter, font, value, name, attrs): # -> None:
        ...
    


class _UInt8Enum(UInt8):
    enumClass = ...
    def read(self, reader, font, tableDict):
        ...
    
    @classmethod
    def fromString(cls, value): # -> Any:
        ...
    
    @classmethod
    def toString(cls, value):
        ...
    


class ExtendMode(_UInt8Enum):
    enumClass = _ExtendMode


class CompositeMode(_UInt8Enum):
    enumClass = _CompositeMode


converterMapping = ...
