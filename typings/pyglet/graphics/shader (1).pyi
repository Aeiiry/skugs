"""
This type stub file was generated by pyright.
"""

from ctypes import *
from pyglet.gl import *

_debug_gl_shaders = ...
class ShaderException(BaseException):
    ...


_c_types = ...
_shader_types = ...
_uniform_getters = ...
_uniform_setters = ...
_attribute_types = ...
class Attribute:
    """Abstract accessor for an attribute in a mapped buffer."""
    def __init__(self, name, location, count, gl_type, normalize) -> None:
        """Create the attribute accessor.

        :Parameters:
            `name` : str
                Name of the vertex attribute.
            `location` : int
                Location (index) of the vertex attribute.
            `count` : int
                Number of components in the attribute.
            `gl_type` : int
                OpenGL type enumerant; for example, ``GL_FLOAT``
            `normalize`: bool
                True if OpenGL should normalize the values

        """
        ...
    
    def enable(self): # -> None:
        """Enable the attribute."""
        ...
    
    def set_pointer(self, ptr): # -> None:
        """Setup this attribute to point to the currently bound buffer at
        the given offset.

        ``offset`` should be based on the currently bound buffer's ``ptr``
        member.

        :Parameters:
            `offset` : int
                Pointer offset to the currently bound buffer for this
                attribute.

        """
        ...
    
    def get_region(self, buffer, start, count):
        """Map a buffer region using this attribute as an accessor.

        The returned region consists of a contiguous array of component
        data elements.  For example, if this attribute uses 3 floats per
        vertex, and the `count` parameter is 4, the number of floats mapped
        will be ``3 * 4 = 12``.

        :Parameters:
            `buffer` : `AbstractMappable`
                The buffer to map.
            `start` : int
                Offset of the first vertex to map.
            `count` : int
                Number of vertices to map

        :rtype: `AbstractBufferRegion`
        """
        ...
    
    def set_region(self, buffer, start, count, data): # -> None:
        """Set the data over a region of the buffer.

        :Parameters:
            `buffer` : AbstractMappable`
                The buffer to modify.
            `start` : int
                Offset of the first vertex to set.
            `count` : int
                Number of vertices to set.
            `data` : A sequence of data components.
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


class _Uniform:
    __slots__ = ...
    def __init__(self, program, name, uniform_type, location, dsa) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class UniformBlock:
    __slots__ = ...
    def __init__(self, program, name, index, size, uniforms) -> None:
        ...
    
    def create_ubo(self, index=...): # -> UniformBufferObject:
        """
        Create a new UniformBufferObject from this uniform block.

        :Parameters:
            `index` : int
                The uniform buffer index the returned UBO will bind itself to.
                By default, this is 0.

        :rtype: :py:class:`~pyglet.graphics.shader.UniformBufferObject`
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


class UniformBufferObject:
    __slots__ = ...
    def __init__(self, view_class, buffer_size, index) -> None:
        ...
    
    @property
    def id(self): # -> int | None:
        ...
    
    def bind(self, index=...): # -> None:
        ...
    
    def read(self): # -> bytes:
        """Read the byte contents of the buffer"""
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb): # -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class ShaderSource:
    """GLSL source container for making source parsing simpler.

    We support locating out attributes and applying #defines values.

    NOTE: We do assume the source is neat enough to be parsed
    this way and don't contain several statements in one line.
    """
    def __init__(self, source: str, source_type: GLenum) -> None:
        """Create a shader source wrapper."""
        ...
    
    def validate(self) -> str:
        """Return the validated shader source."""
        ...
    


class Shader:
    """OpenGL Shader object"""
    def __init__(self, source_string: str, shader_type: str) -> None:
        """Create an instance of a Shader object.

        Shader source code should be provided as a Python string.
        The shader_type should be a string indicating the type.
        Valid types are: 'compute', 'fragment', 'geometry', 'tesscontrol',
        'tessevaluation', and 'vertex'.

        Shader objects are compiled on instantiation.
        You can reuse a Shader object in multiple `ShaderProgram`s.
        """
        ...
    
    @property
    def id(self): # -> Any | None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


class ShaderProgram:
    """OpenGL Shader Program"""
    __slots__ = ...
    def __init__(self, *shaders: Shader) -> None:
        """Create an OpenGL ShaderProgram from one or more Shader objects."""
        ...
    
    @property
    def id(self): # -> int:
        ...
    
    @property
    def attributes(self): # -> dict[Unknown, Unknown]:
        ...
    
    @property
    def uniforms(self): # -> dict[Unknown, Unknown]:
        ...
    
    @property
    def uniform_blocks(self): # -> dict[Unknown, Unknown]:
        ...
    
    def use(self): # -> None:
        ...
    
    @staticmethod
    def stop(): # -> None:
        ...
    
    __enter__ = ...
    bind = ...
    unbind = ...
    def __exit__(self, *_): # -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __setitem__(self, key, value): # -> None:
        ...
    
    def __getitem__(self, item):
        ...
    
    def vertex_list(self, count, mode, batch=..., group=..., **data): # -> VertexList:
        """Create a VertexList.

        :Parameters:
            `count` : int
                The number of vertices in the list.
            `mode` : int
                OpenGL drawing mode enumeration; for example, one of
                ``GL_POINTS``, ``GL_LINES``, ``GL_TRIANGLES``, etc.
            `batch` : `~pyglet.graphics.Batch`
                Batch to add the VertexList to, or ``None`` if a Batch will not be used.
                Using a Batch is strongly recommended.
            `group` : `~pyglet.graphics.Group`
                Group to add the VertexList to, or ``None`` if no group is required.
            `**data` : str or tuple
                Attribute formats and initial data for the vertex list.

        :rtype: :py:class:`~pyglet.graphics.vertexdomain.VertexList`
        """
        ...
    
    def vertex_list_indexed(self, count, mode, indices, batch=..., group=..., **data): # -> IndexedVertexList | VertexList:
        """Create a IndexedVertexList.

        :Parameters:
            `count` : int
                The number of vertices in the list.
            `mode` : int
                OpenGL drawing mode enumeration; for example, one of
                ``GL_POINTS``, ``GL_LINES``, ``GL_TRIANGLES``, etc.
            `indices` : sequence of int
                Sequence of integers giving indices into the vertex list.
            `batch` : `~pyglet.graphics.Batch`
                Batch to add the VertexList to, or ``None`` if a Batch will not be used.
                Using a Batch is strongly recommended.
            `group` : `~pyglet.graphics.Group`
                Group to add the VertexList to, or ``None`` if no group is required.
            `**data` : str or tuple
                Attribute formats and initial data for the vertex list.

        :rtype: :py:class:`~pyglet.graphics.vertexdomain.IndexedVertexList`
        """
        ...
    
    def __repr__(self): # -> str:
        ...
    


class ComputeShaderProgram:
    """OpenGL Compute Shader Program"""
    __slots__ = ...
    def __init__(self, source: str) -> None:
        """Create an OpenGL ComputeShaderProgram from source."""
        ...
    
    @staticmethod
    def dispatch(x: int = ..., y: int = ..., z: int = ..., barrier: int = ...) -> None:
        """Launch one or more compute work groups.

        The ComputeShaderProgram should be active (bound) before calling
        this method. The x, y, and z parameters specify the number of local
        work groups that will be  dispatched in the X, Y and Z dimensions.
        """
        ...
    
    @property
    def id(self) -> int:
        ...
    
    @property
    def uniforms(self) -> dict:
        ...
    
    @property
    def uniform_blocks(self) -> dict:
        ...
    
    def use(self) -> None:
        ...
    
    @staticmethod
    def stop(): # -> None:
        ...
    
    __enter__ = ...
    bind = ...
    unbind = ...
    def __exit__(self, *_): # -> None:
        ...
    
    def __del__(self): # -> None:
        ...
    
    def __setitem__(self, key, value): # -> None:
        ...
    
    def __getitem__(self, item):
        ...
    


