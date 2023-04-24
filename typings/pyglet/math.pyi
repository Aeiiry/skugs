"""
This type stub file was generated by pyright.
"""

import typing as _typing
from collections.abc import Iterable as _Iterable, Iterator as _Iterator

"""Matrix and Vector math.

This module provides Vector and Matrix objects, including Vec2, Vec3,
Vec4, Mat3, and Mat4. Most common matrix and vector operations are
supported. Helper methods are included for rotating, scaling, and
transforming. The :py:class:`~pyglet.matrix.Mat4` includes class methods
for creating orthographic and perspective projection matrixes.

Matrices behave just like they do in GLSL: they are specified in column-major
order and multiply on the left of vectors, which are treated as columns.

:note: For performance, Matrixes subclass the `tuple` type. They
    are therefore immutable - all operations return a new object;
    the object is not updated in-place.
"""
number = _typing.Union[float, int]
Mat4T = _typing.TypeVar("Mat4T", bound="Mat4")
def clamp(num: float, min_val: float, max_val: float) -> float:
    ...

class Vec2:
    __slots__ = ...
    def __init__(self, x: number = ..., y: number = ...) -> None:
        ...
    
    def __iter__(self) -> _Iterator[float]:
        ...
    
    @_typing.overload
    def __getitem__(self, item: int) -> float:
        ...
    
    @_typing.overload
    def __getitem__(self, item: slice) -> tuple[float, ...]:
        ...
    
    def __getitem__(self, item):
        ...
    
    def __setitem__(self, key, value): # -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __add__(self, other: Vec2) -> Vec2:
        ...
    
    def __sub__(self, other: Vec2) -> Vec2:
        ...
    
    def __mul__(self, scalar: number) -> Vec2:
        ...
    
    def __truediv__(self, scalar: number) -> Vec2:
        ...
    
    def __floordiv__(self, scalar: number) -> Vec2:
        ...
    
    def __abs__(self) -> float:
        ...
    
    def __neg__(self) -> Vec2:
        ...
    
    def __round__(self, ndigits: int | None = ...) -> Vec2:
        ...
    
    def __radd__(self, other: Vec2 | int) -> Vec2:
        """Reverse add. Required for functionality with sum()
        """
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    @staticmethod
    def from_polar(mag: float, angle: float) -> Vec2:
        """Create a new vector from the given polar coordinates.

        :parameters:
            `mag`   : int or float :
                The magnitude of the vector.
            `angle` : int or float :
                The angle of the vector in radians.

        :returns: A new vector with the given angle and magnitude.
        :rtype: Vec2
        """
        ...
    
    def from_magnitude(self, magnitude: float) -> Vec2:
        """Create a new Vector of the given magnitude by normalizing,
        then scaling the vector. The heading remains unchanged.

        :parameters:
            `magnitude` : int or float :
                The magnitude of the new vector.

        :returns: A new vector with the magnitude.
        :rtype: Vec2
        """
        ...
    
    def from_heading(self, heading: float) -> Vec2:
        """Create a new vector of the same magnitude with the given heading. I.e. Rotate the vector to the heading.

        :parameters:
            `heading` : int or float :
                The angle of the new vector in radians.

        :returns: A new vector with the given heading.
        :rtype: Vec2
        """
        ...
    
    @property
    def heading(self) -> float:
        """The angle of the vector in radians.

        :type: float
        """
        ...
    
    @property
    def mag(self) -> float:
        """The magnitude, or length of the vector. The distance between the coordinates and the origin.

        Alias of abs(self).

        :type: float
        """
        ...
    
    def limit(self, maximum: float) -> Vec2:
        """Limit the magnitude of the vector to the value used for the max parameter.

        :parameters:
            `maximum`  : int or float :
                The maximum magnitude for the vector.

        :returns: Either self or a new vector with the maximum magnitude.
        :rtype: Vec2
        """
        ...
    
    def lerp(self, other: Vec2, alpha: float) -> Vec2:
        """Create a new Vec2 linearly interpolated between this vector and another Vec2.

        :parameters:
            `other`  : Vec2 :
                The vector to linearly interpolate with.
            `alpha` : float or int :
                The amount of interpolation.
                Some value between 0.0 (this vector) and 1.0 (other vector).
                0.5 is halfway inbetween.

        :returns: A new interpolated vector.
        :rtype: Vec2
        """
        ...
    
    def reflect(self, normal: Vec2) -> Vec2:
        """Create a new Vec2 reflected (ricochet) from the given normal."""
        ...
    
    def rotate(self, angle: float) -> Vec2:
        """Create a new Vector rotated by the angle. The magnitude remains unchanged.

        :parameters:
            `angle` : int or float :
                The angle to rotate by

        :returns: A new rotated vector of the same magnitude.
        :rtype: Vec2
        """
        ...
    
    def distance(self, other: Vec2) -> float:
        """Calculate the distance between this vector and another 2D vector."""
        ...
    
    def normalize(self) -> Vec2:
        """Normalize the vector to have a magnitude of 1. i.e. make it a unit vector.

        :returns: A unit vector with the same heading.
        :rtype: Vec2
        """
        ...
    
    def clamp(self, min_val: float, max_val: float) -> Vec2:
        """Restrict the value of the X and Y components of the vector to be within the given values.

        :parameters:
            `min_val` : int or float :
                The minimum value
            `max_val` : int or float :
                The maximum value

        :returns: A new vector with clamped X and Y components.
        :rtype: Vec2
        """
        ...
    
    def dot(self, other: Vec2) -> float:
        """Calculate the dot product of this vector and another 2D vector.

        :parameters:
            `other`  : Vec2 :
                The other vector.

        :returns: The dot product of the two vectors.
        :rtype: float
        """
        ...
    
    def __getattr__(self, attrs: str) -> Vec2 | Vec3 | Vec4:
        ...
    
    def __repr__(self) -> str:
        ...
    


class Vec3:
    __slots__ = ...
    def __init__(self, x: number = ..., y: number = ..., z: number = ...) -> None:
        ...
    
    def __iter__(self) -> _Iterator[float]:
        ...
    
    @_typing.overload
    def __getitem__(self, item: int) -> float:
        ...
    
    @_typing.overload
    def __getitem__(self, item: slice) -> tuple[float, ...]:
        ...
    
    def __getitem__(self, item):
        ...
    
    def __setitem__(self, key, value): # -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    @property
    def mag(self) -> float:
        """The magnitude, or length of the vector. The distance between the coordinates and the origin.

        Alias of abs(self).

        :type: float
        """
        ...
    
    def __add__(self, other: Vec3) -> Vec3:
        ...
    
    def __sub__(self, other: Vec3) -> Vec3:
        ...
    
    def __mul__(self, scalar: number) -> Vec3:
        ...
    
    def __truediv__(self, scalar: number) -> Vec3:
        ...
    
    def __floordiv__(self, scalar: number) -> Vec3:
        ...
    
    def __abs__(self) -> float:
        ...
    
    def __neg__(self) -> Vec3:
        ...
    
    def __round__(self, ndigits: int | None = ...) -> Vec3:
        ...
    
    def __radd__(self, other: Vec3 | int) -> Vec3:
        """Reverse add. Required for functionality with sum()"""
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def from_magnitude(self, magnitude: float) -> Vec3:
        """Create a new Vector of the given magnitude by normalizing,
        then scaling the vector. The rotation remains unchanged.

        :parameters:
            `magnitude` : int or float :
                The magnitude of the new vector.

        :returns: A new vector with the magnitude.
        :rtype: Vec3
        """
        ...
    
    def limit(self, maximum: float) -> Vec3:
        """Limit the magnitude of the vector to the value used for the max parameter.

        :parameters:
            `maximum`  : int or float :
                The maximum magnitude for the vector.

        :returns: Either self or a new vector with the maximum magnitude.
        :rtype: Vec3
        """
        ...
    
    def cross(self, other: Vec3) -> Vec3:
        """Calculate the cross product of this vector and another 3D vector.

        :parameters:
            `other`  : Vec3 :
                The other vector.

        :returns: The cross product of the two vectors.
        :rtype: float
        """
        ...
    
    def dot(self, other: Vec3) -> float:
        """Calculate the dot product of this vector and another 3D vector.

        :parameters:
            `other`  : Vec3 :
                The other vector.

        :returns: The dot product of the two vectors.
        :rtype: float
        """
        ...
    
    def lerp(self, other: Vec3, alpha: float) -> Vec3:
        """Create a new Vec3 linearly interpolated between this vector and another Vec3.

        :parameters:
            `other`  : Vec3 :
                The vector to linearly interpolate with.
            `alpha` : float or int :
                The amount of interpolation.
                Some value between 0.0 (this vector) and 1.0 (other vector).
                0.5 is halfway inbetween.

        :returns: A new interpolated vector.
        :rtype: Vec3
        """
        ...
    
    def distance(self, other: Vec3) -> float:
        """Calculate the distance between this vector and another 3D vector.

        :parameters:
            `other`  : Vec3 :
                The other vector

        :returns: The distance between the two vectors.
        :rtype: float
        """
        ...
    
    def normalize(self) -> Vec3:
        """Normalize the vector to have a magnitude of 1. i.e. make it a unit vector.

        :returns: A unit vector with the same rotation.
        :rtype: Vec3
        """
        ...
    
    def clamp(self, min_val: float, max_val: float) -> Vec3:
        """Restrict the value of the X,  Y and Z components of the vector to be within the given values.

        :parameters:
            `min_val` : int or float :
                The minimum value
            `max_val` : int or float :
                The maximum value

        :returns: A new vector with clamped X, Y and Z components.
        :rtype: Vec3
        """
        ...
    
    def __getattr__(self, attrs: str) -> Vec2 | Vec3 | Vec4:
        ...
    
    def __repr__(self) -> str:
        ...
    


class Vec4:
    __slots__ = ...
    def __init__(self, x: number = ..., y: number = ..., z: number = ..., w: number = ...) -> None:
        ...
    
    def __iter__(self) -> _Iterator[float]:
        ...
    
    @_typing.overload
    def __getitem__(self, item: int) -> float:
        ...
    
    @_typing.overload
    def __getitem__(self, item: slice) -> tuple[float, ...]:
        ...
    
    def __getitem__(self, item):
        ...
    
    def __setitem__(self, key, value): # -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __add__(self, other: Vec4) -> Vec4:
        ...
    
    def __sub__(self, other: Vec4) -> Vec4:
        ...
    
    def __mul__(self, scalar: number) -> Vec4:
        ...
    
    def __truediv__(self, scalar: number) -> Vec4:
        ...
    
    def __floordiv__(self, scalar: number) -> Vec4:
        ...
    
    def __abs__(self) -> float:
        ...
    
    def __neg__(self) -> Vec4:
        ...
    
    def __round__(self, ndigits: int | None = ...) -> Vec4:
        ...
    
    def __radd__(self, other: Vec4 | int) -> Vec4:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __ne__(self, other: object) -> bool:
        ...
    
    def lerp(self, other: Vec4, alpha: float) -> Vec4:
        """Create a new Vec4 linearly interpolated between this one and another Vec4.

        :parameters:
            `other`  : Vec4 :
                The vector to linearly interpolate with.
            `alpha` : float or int :
                The amount of interpolation.
                Some value between 0.0 (this vector) and 1.0 (other vector).
                0.5 is halfway inbetween.

        :returns: A new interpolated vector.
        :rtype: Vec4
        """
        ...
    
    def distance(self, other: Vec4) -> float:
        ...
    
    def normalize(self) -> Vec4:
        """Normalize the vector to have a magnitude of 1. i.e. make it a unit vector."""
        ...
    
    def clamp(self, min_val: float, max_val: float) -> Vec4:
        ...
    
    def dot(self, other: Vec4) -> float:
        ...
    
    def __getattr__(self, attrs: str) -> Vec2 | Vec3 | Vec4:
        ...
    
    def __repr__(self) -> str:
        ...
    


class Mat3(tuple):
    """A 3x3 Matrix class

    `Mat3` is an immutable 3x3 Matrix, including most common
    operators. Matrix multiplication must be performed using
    the "@" operator.
    """
    def __new__(cls, values: _Iterable[float] = ...) -> Mat3:
        """Create a 3x3 Matrix

        A Mat3 can be created with a list or tuple of 9 values.
        If no values are provided, an "identity matrix" will be created
        (1.0 on the main diagonal). Matrix objects are immutable, so
        all operations return a new Mat3 object.

        :Parameters:
            `values` : tuple of float or int
                A tuple or list containing 9 floats or ints.
        """
        ...
    
    def scale(self, sx: float, sy: float) -> Mat3:
        ...
    
    def translate(self, tx: float, ty: float) -> Mat3:
        ...
    
    def rotate(self, phi: float) -> Mat3:
        ...
    
    def shear(self, sx: float, sy: float) -> Mat3:
        ...
    
    def __add__(self, other: Mat3) -> Mat3:
        ...
    
    def __sub__(self, other: Mat3) -> Mat3:
        ...
    
    def __pos__(self) -> Mat3:
        ...
    
    def __neg__(self) -> Mat3:
        ...
    
    def __round__(self, ndigits: int | None = ...) -> Mat3:
        ...
    
    def __mul__(self, other: object) -> _typing.NoReturn:
        ...
    
    @_typing.overload
    def __matmul__(self, other: Vec3) -> Vec3:
        ...
    
    @_typing.overload
    def __matmul__(self, other: Mat3) -> Mat3:
        ...
    
    def __matmul__(self, other): # -> Vec3 | Mat3:
        ...
    
    def __repr__(self) -> str:
        ...
    


class Mat4(tuple):
    """A 4x4 Matrix class

    `Mat4` is an immutable 4x4 Matrix, including most common
    operators. Matrix multiplication must be performed using
    the "@" operator.
    Class methods are available for creating orthogonal
    and perspective projections matrixes.
    """
    def __new__(cls, values: _Iterable[float] = ...) -> Mat4:
        """Create a 4x4 Matrix

        A Matrix can be created with a list or tuple of 16 values.
        If no values are provided, an "identity matrix" will be created
        (1.0 on the main diagonal). Matrix objects are immutable, so
        all operations return a new Mat4 object.

        :Parameters:
            `values` : tuple of float or int
                A tuple or list containing 16 floats or ints.
        """
        ...
    
    @classmethod
    def orthogonal_projection(cls: type[Mat4T], left: float, right: float, bottom: float, top: float, z_near: float, z_far: float) -> Mat4T:
        """Create a Mat4 orthographic projection matrix for use with OpenGL.

        This matrix doesn't actually perform the projection; it transforms the
        space so that OpenGL's vertex processing performs it.
        """
        ...
    
    @classmethod
    def perspective_projection(cls: type[Mat4T], aspect: float, z_near: float, z_far: float, fov: float = ...) -> Mat4T:
        """
        Create a Mat4 perspective projection matrix for use with OpenGL.

        This matrix doesn't actually perform the projection; it transforms the
        space so that OpenGL's vertex processing performs it.

        :Parameters:
            `aspect` : The aspect ratio as a `float`
            `z_near` : The near plane as a `float`
            `z_far` : The far plane as a `float`
            `fov` : Field of view in degrees as a `float`
        """
        ...
    
    @classmethod
    def from_rotation(cls, angle: float, vector: Vec3) -> Mat4:
        """Create a rotation matrix from an angle and Vec3.

        :Parameters:
            `angle` : A `float` :
                The angle as a float.
            `vector` : A `Vec3`, or 3 component tuple of float or int :
                Vec3 or tuple with x, y and z translation values
        """
        ...
    
    @classmethod
    def from_scale(cls: type[Mat4T], vector: Vec3) -> Mat4T:
        """Create a scale matrix from a Vec3.

        :Parameters:
            `vector` : A `Vec3`, or 3 component tuple of float or int
                Vec3 or tuple with x, y and z scale values
        """
        ...
    
    @classmethod
    def from_translation(cls: type[Mat4T], vector: Vec3) -> Mat4T:
        """Create a translation matrix from a Vec3.

        :Parameters:
            `vector` : A `Vec3`, or 3 component tuple of float or int
                Vec3 or tuple with x, y and z translation values
        """
        ...
    
    @classmethod
    def look_at(cls: type[Mat4T], position: Vec3, target: Vec3, up: Vec3): # -> Mat4T@look_at:
        ...
    
    def row(self, index: int) -> tuple:
        """Get a specific row as a tuple."""
        ...
    
    def column(self, index: int) -> tuple:
        """Get a specific column as a tuple."""
        ...
    
    def rotate(self, angle: float, vector: Vec3) -> Mat4:
        """Get a rotation Matrix on x, y, or z axis."""
        ...
    
    def scale(self, vector: Vec3) -> Mat4:
        """Get a scale Matrix on x, y, or z axis."""
        ...
    
    def translate(self, vector: Vec3) -> Mat4:
        """Get a translation Matrix along x, y, and z axis."""
        ...
    
    def transpose(self) -> Mat4:
        """Get a transpose of this Matrix."""
        ...
    
    def __add__(self, other: Mat4) -> Mat4:
        ...
    
    def __sub__(self, other: Mat4) -> Mat4:
        ...
    
    def __pos__(self) -> Mat4:
        ...
    
    def __neg__(self) -> Mat4:
        ...
    
    def __invert__(self) -> Mat4:
        ...
    
    def __round__(self, ndigits: int | None = ...) -> Mat4:
        ...
    
    def __mul__(self, other: int) -> _typing.NoReturn:
        ...
    
    @_typing.overload
    def __matmul__(self, other: Vec4) -> Vec4:
        ...
    
    @_typing.overload
    def __matmul__(self, other: Mat4) -> Mat4:
        ...
    
    def __matmul__(self, other): # -> Vec4 | Mat4:
        ...
    
    def __repr__(self) -> str:
        ...
    


