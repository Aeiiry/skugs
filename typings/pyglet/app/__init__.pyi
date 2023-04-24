"""
This type stub file was generated by pyright.
"""

import sys
import weakref
from pyglet.app.base import EventLoop, PlatformEventLoop
from pyglet import compat_platform

"""Application-wide functionality.

Applications
------------

Most applications need only call :func:`run` after creating one or more 
windows to begin processing events.  For example, a simple application 
consisting of one window is::

    import pyglet

    win = pyglet.window.Window()
    pyglet.app.run()


Events
======

To handle events on the main event loop, instantiate it manually.  The
following example exits the application as soon as any window is closed (the
default policy is to wait until all windows are closed)::

    event_loop = pyglet.app.EventLoop()

    @event_loop.event
    def on_window_close(window):
        event_loop.exit()

.. versionadded:: 1.1
"""
_is_pyglet_doc_run = ...
if _is_pyglet_doc_run:
    ...
else:
    ...
class AppException(Exception):
    ...


windows = ...
def run(interval=...): # -> None:
    """Begin processing events, scheduled functions and window updates.

    This is a convenience function, equivalent to::

        pyglet.app.event_loop.run()

    """
    ...

def exit(): # -> None:
    """Exit the application event loop.

    Causes the application event loop to finish, if an event loop is currently
    running.  The application may not necessarily exit (for example, there may
    be additional code following the `run` invocation).

    This is a convenience function, equivalent to::

        event_loop.exit()

    """
    ...

event_loop = ...
platform_event_loop = ...
