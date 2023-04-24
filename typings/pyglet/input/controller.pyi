"""
This type stub file was generated by pyright.
"""

"""Game Controller support.

This module provides an interface for Game Controller devices, which are a
subset of Joysticks. Game Controllers have consistent button and axis mapping,
which resembles common dual-stick home video game console controllers.
Devices that are of this design can be automatically mapped to the "virtual"
Game Controller layout, providing a consistent abstraction for a large number
of different devices, with no tedious button and axis mapping for each one.
To achieve this, an internal mapping database contains lists of device ids
and their corresponding button and axis mappings. The mapping database is in
the same format as originated by the `SDL` library, which has become a
semi-standard and is in common use. Most popular controllers are included in
the built-in database, and additional mappings can be added at runtime.


Some Joysticks, such as Flight Sticks, etc., do not necessarily fit into the
layout (and limitations) of GameControllers. For those such devices, it is
recommended to use the Joystick interface instead.

To query which GameControllers are available, call :py:func:`get_controllers`.

.. versionadded:: 2.0
"""
_env_config = ...
if _env_config:
    ...
def create_guid(bus: int, vendor: int, product: int, version: int, name: str, signature: int, data: int) -> str:
    """Create an SDL2 style GUID string from a device's identifiers."""
    ...

class Relation:
    __slots__ = ...
    def __init__(self, control_type, index, inverted=...) -> None:
        ...
    
    def __repr__(self): # -> str:
        ...
    


def get_mapping(guid): # -> dict[str, Unknown] | None:
    """Return a mapping for the passed device GUID.

    :Parameters:
        `guid` : str
            A pyglet input device GUID

    :rtype: dict of axis/button mapping relations, or None
            if no mapping is available for this Controller.
    """
    ...

def add_mappings_from_file(filename) -> None:
    """Add mappings from a file.

    Given a file path, open and parse the file for mappings.

    :Parameters:
        `filename` : str
            A file path.
    """
    ...

def add_mappings_from_string(string) -> None:
    """Add one or more mappings from a raw string.

        :Parameters:
            `string` : str
                A string containing one or more mappings,
        """
    ...

