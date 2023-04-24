"""
This type stub file was generated by pyright.
"""

from . import DefaultTable

class table_V_O_R_G_(DefaultTable.DefaultTable):
    """This table is structured so that you can treat it like a dictionary keyed by glyph name.

    ``ttFont['VORG'][<glyphName>]`` will return the vertical origin for any glyph.

    ``ttFont['VORG'][<glyphName>] = <value>`` will set the vertical origin for any glyph.
    """
    def decompile(self, data, ttFont): # -> None:
        ...
    
    def compile(self, ttFont): # -> bytes:
        ...
    
    def toXML(self, writer, ttFont): # -> None:
        ...
    
    def fromXML(self, name, attrs, content, ttFont): # -> None:
        ...
    
    def __getitem__(self, glyphSelector): # -> Any:
        ...
    
    def __setitem__(self, glyphSelector, value): # -> None:
        ...
    
    def __delitem__(self, glyphSelector): # -> None:
        ...
    


class VOriginRecord:
    def __init__(self, name=..., vOrigin=...) -> None:
        ...
    
    def toXML(self, writer, ttFont): # -> None:
        ...
    
    def fromXML(self, name, attrs, content, ttFont): # -> None:
        ...
    


