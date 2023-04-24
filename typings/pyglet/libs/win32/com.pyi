"""
This type stub file was generated by pyright.
"""

import sys
import ctypes

"""Minimal Windows COM interface.

Allows pyglet to use COM interfaces on Windows without comtypes.  Unlike
comtypes, this module does not provide property interfaces, read typelibs,
nice-ify return values.  We don't need anything that sophisticated to work with COM's.

Interfaces should derive from pIUnknown if their implementation is returned by the COM.
The Python COM interfaces are actually pointers to the implementation (take note
when translating methods that take an interface as argument).
(example: A Double Pointer is simply POINTER(MyInterface) as pInterface is already a POINTER.)

Interfaces can define methods::

    class IDirectSound8(com.pIUnknown):
        _methods_ = [
            ('CreateSoundBuffer', com.STDMETHOD()),
            ('GetCaps', com.STDMETHOD(LPDSCAPS)),
            ...
        ]

Only use STDMETHOD or METHOD for the method types (not ordinary ctypes
function types).  The 'this' pointer is bound automatically... e.g., call::

    device = IDirectSound8()
    DirectSoundCreate8(None, ctypes.byref(device), None)

    caps = DSCAPS()
    device.GetCaps(caps)

Because STDMETHODs use HRESULT as the return type, there is no need to check
the return value.

Don't forget to manually manage memory... call Release() when you're done with
an interface.
"""
_debug_com = ...
if sys.platform != 'win32':
    ...
class GUID(ctypes.Structure):
    _fields_ = ...
    def __init__(self, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) -> None:
        ...
    
    def __repr__(self): # -> LiteralString:
        ...
    
    def __cmp__(self, other): # -> Literal[-1]:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    


LPGUID = ...
IID = GUID
REFIID = ...
class METHOD:
    """COM method."""
    def __init__(self, restype, *args) -> None:
        ...
    
    def get_field(self): # -> Type[_FuncPointer]:
        ...
    


class STDMETHOD(METHOD):
    """COM method with HRESULT return value."""
    def __init__(self, *args) -> None:
        ...
    


class COMMethodInstance:
    """Binds a COM interface method."""
    def __init__(self, name, i, method) -> None:
        ...
    
    def __get__(self, obj, tp): # -> (*args: Unknown) -> Unknown:
        ...
    


class COMInterface(ctypes.Structure):
    """Dummy struct to serve as the type of all COM pointers."""
    _fields_ = ...


class InterfacePtrMeta(type(ctypes.POINTER(COMInterface))):
    """Allows interfaces to be subclassed as ctypes POINTER and expects to be populated with data from a COM object.
       TODO: Phase this out and properly use POINTER(Interface) where applicable.
    """
    def __new__(cls, name, bases, dct):
        ...
    


pInterface = ...
class COMInterfaceMeta(type):
    """This differs in the original as an implemented interface object, not a POINTER object.
       Used when the user must implement their own functions within an interface rather than
       being created and generated by the COM object itself. The types are automatically inserted in the ctypes type
       cache so it can recognize the type arguments.
    """
    def __new__(mcs, name, bases, dct): # -> Self@COMInterfaceMeta:
        ...
    


class COMPointerMeta(type(ctypes.c_void_p), COMInterfaceMeta):
    """Required to prevent metaclass conflicts with inheritance."""
    ...


class COMPointer(ctypes.c_void_p, metaclass=COMPointerMeta):
    """COM Pointer base, could use c_void_p but need to override from_param ."""
    @classmethod
    def from_param(cls, obj): # -> Any | None:
        """Allows obj to return ctypes pointers, even if its base is not a ctype.
           In this case, all we simply want is a ctypes pointer matching the cls interface from the obj.
        """
        ...
    


_cached_structures = ...
def create_vtbl_structure(fields, interface): # -> Type[_]:
    """Create virtual table structure with fields for use in COM's."""
    ...

class COMObject:
    """A base class for defining a COM object for use with callbacks and custom implementations."""
    _interfaces_ = ...
    def __new__(cls, *args, **kw): # -> Self@COMObject:
        ...
    
    @property
    def pointers(self):
        """Returns pointers to the implemented interfaces in this COMObject.  Read-only.

        :type: dict
        """
        ...
    


class Interface(metaclass=COMInterfaceMeta):
    _methods_ = ...


class IUnknown(metaclass=COMInterfaceMeta):
    """These methods are not implemented by default yet. Strictly for COM method ordering."""
    _methods_ = ...


class pIUnknown(pInterface):
    _methods_ = ...


