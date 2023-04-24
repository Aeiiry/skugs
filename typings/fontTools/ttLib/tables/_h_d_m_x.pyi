"""
This type stub file was generated by pyright.
"""

from . import DefaultTable
from collections.abc import Mapping

hdmxHeaderFormat = ...
class _GlyphnamedList(Mapping):
    def __init__(self, reverseGlyphOrder, data) -> None:
        ...
    
    def __getitem__(self, k):
        ...
    
    def __len__(self): # -> int:
        ...
    
    def __iter__(self): # -> Iterator[Unknown]:
        ...
    
    def keys(self): # -> dict_keys[Unknown, Unknown]:
        ...
    


class table__h_d_m_x(DefaultTable.DefaultTable):
    def decompile(self, data, ttFont): # -> None:
        ...
    
    def compile(self, ttFont): # -> bytes:
        ...
    
    def toXML(self, writer, ttFont): # -> None:
        ...
    
    def fromXML(self, name, attrs, content, ttFont): # -> None:
        ...
    


