"""
This type stub file was generated by pyright.
"""

from . import DefaultTable

Sill_hdr = ...
class table_S__i_l_l(DefaultTable.DefaultTable):
    def __init__(self, tag=...) -> None:
        ...
    
    def decompile(self, data, ttFont): # -> None:
        ...
    
    def compile(self, ttFont): # -> bytes:
        ...
    
    def toXML(self, writer, ttFont): # -> None:
        ...
    
    def fromXML(self, name, attrs, content, ttFont): # -> None:
        ...
    


