"""
This type stub file was generated by pyright.
"""

from . import DefaultTable

Feat_hdr_format = ...
class table_F__e_a_t(DefaultTable.DefaultTable):
    """The ``Feat`` table is used exclusively by the Graphite shaping engine
    to store features and possible settings specified in GDL. Graphite features
    determine what rules are applied to transform a glyph stream.

    Not to be confused with ``feat``, or the OpenType Layout tables
    ``GSUB``/``GPOS``."""
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
    


class Feature:
    ...


