"""
This type stub file was generated by pyright.
"""

__all__ = ['VertexArray']
class VertexArray:
    """OpenGL Vertex Array Object"""
    def __init__(self) -> None:
        """Create an instance of a Vertex Array object."""
        ...
    
    @property
    def id(self): # -> int:
        ...
    
    def bind(self): # -> None:
        ...
    
    @staticmethod
    def unbind(): # -> None:
        ...
    
    def delete(self): # -> None:
        ...
    
    __enter__ = ...
    def __exit__(self, *_): # -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


