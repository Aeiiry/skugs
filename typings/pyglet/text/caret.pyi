"""
This type stub file was generated by pyright.
"""

"""Provides keyboard and mouse editing procedures for text layout.

Example usage::

    from pyglet import window
    from pyglet.text import layout, caret

    my_window = window.Window(...)
    my_layout = layout.IncrementalTextLayout(...)
    my_caret = caret.Caret(my_layout)
    my_window.push_handlers(my_caret)

.. versionadded:: 1.1
"""
class Caret:
    """Visible text insertion marker for
    `pyglet.text.layout.IncrementalTextLayout`.

    The caret is drawn as a single vertical bar at the document `position` 
    on a text layout object.  If `mark` is not None, it gives the unmoving
    end of the current text selection.  The visible text selection on the
    layout is updated along with `mark` and `position`.
    
    By default the layout's graphics batch is used, so the caret does not need
    to be drawn explicitly.  Even if a different graphics batch is supplied,
    the caret will be correctly positioned and clipped within the layout.

    Updates to the document (and so the layout) are automatically propagated
    to the caret.  

    The caret object can be pushed onto a window event handler stack with
    `Window.push_handlers`.  The caret will respond correctly to keyboard,
    text, mouse and activation events, including double- and triple-clicks.
    If the text layout is being used alongside other graphical widgets, a
    GUI toolkit will be needed to delegate keyboard and mouse events to the
    appropriate widget.  pyglet does not provide such a toolkit at this stage.
    """
    _next_word_re = ...
    _previous_word_re = ...
    _next_para_re = ...
    _previous_para_re = ...
    _position = ...
    _active = ...
    _visible = ...
    _blink_visible = ...
    _click_count = ...
    _click_time = ...
    PERIOD = ...
    SCROLL_INCREMENT = ...
    _mark = ...
    def __init__(self, layout, batch=..., color=...) -> None:
        """Create a caret for a layout.

        By default the layout's batch is used, so the caret does not need to
        be drawn explicitly.

        :Parameters:
            `layout` : `~pyglet.text.layout.TextLayout`
                Layout to control.
            `batch` : `~pyglet.graphics.Batch`
                Graphics batch to add vertices to.
            `color` : (int, int, int)
                RGB tuple with components in range [0, 255].

        """
        ...
    
    def delete(self): # -> None:
        """Remove the caret from its batch.

        Also disconnects the caret from further layout events.
        """
        ...
    
    @property
    def visible(self): # -> bool:
        """Caret visibility.

        The caret may be hidden despite this property due to the periodic blinking
        or by `on_deactivate` if the event handler is attached to a window.

        :type: bool
        """
        ...
    
    @visible.setter
    def visible(self, visible): # -> None:
        ...
    
    @property
    def color(self):
        """Caret color.

        The default caret color is ``[0, 0, 0]`` (black).  Each RGB color
        component is in the range 0 to 255.

        :type: (int, int, int)
        """
        ...
    
    @color.setter
    def color(self, color): # -> None:
        ...
    
    @property
    def position(self): # -> int:
        """Position of caret within document."""
        ...
    
    @position.setter
    def position(self, position): # -> None:
        ...
    
    @property
    def mark(self): # -> None:
        """Position of immovable end of text selection within document.

        An interactive text selection is determined by its immovable end (the
        caret's position when a mouse drag begins) and the caret's position, which
        moves interactively by mouse and keyboard input.

        This property is ``None`` when there is no selection.

        :type: int
        """
        ...
    
    @mark.setter
    def mark(self, mark): # -> None:
        ...
    
    @property
    def line(self):
        """Index of line containing the caret's position.

        When set, `position` is modified to place the caret on requested line
        while maintaining the closest possible X offset.

        :rtype: int
        """
        ...
    
    @line.setter
    def line(self, line): # -> None:
        ...
    
    def get_style(self, attribute):
        """Get the document's named style at the caret's current position.

        If there is a text selection and the style varies over the selection,
        `pyglet.text.document.STYLE_INDETERMINATE` is returned.

        :Parameters:
            `attribute` : str
                Name of style attribute to retrieve.  See
                `pyglet.text.document` for a list of recognised attribute
                names.

        :rtype: object
        """
        ...
    
    def set_style(self, attributes): # -> None:
        """Set the document style at the caret's current position.

        If there is a text selection the style is modified immediately.
        Otherwise, the next text that is entered before the position is
        modified will take on the given style.

        :Parameters:
            `attributes` : dict
                Dict mapping attribute names to style values.  See
                `pyglet.text.document` for a list of recognised attribute
                names.

        """
        ...
    
    def move_to_point(self, x, y): # -> None:
        """Move the caret close to the given window coordinate.

        The `mark` will be reset to ``None``.

        :Parameters:
            `x` : int   
                X coordinate.
            `y` : int
                Y coordinate.

        """
        ...
    
    def select_to_point(self, x, y): # -> None:
        """Move the caret close to the given window coordinate while
        maintaining the `mark`.

        :Parameters:
            `x` : int   
                X coordinate.
            `y` : int
                Y coordinate.

        """
        ...
    
    def select_word(self, x, y): # -> None:
        """Select the word at the given window coordinate.

        :Parameters:
            `x` : int   
                X coordinate.
            `y` : int
                Y coordinate.

        """
        ...
    
    def select_paragraph(self, x, y): # -> None:
        """Select the paragraph at the given window coordinate.

        :Parameters:
            `x` : int   
                X coordinate.
            `y` : int
                Y coordinate.

        """
        ...
    
    def on_layout_update(self): # -> None:
        """Handler for the `IncrementalTextLayout.on_layout_update` event.
        """
        ...
    
    def on_text(self, text): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_text` event.

        Caret keyboard handlers assume the layout always has keyboard focus.
        GUI toolkits should filter keyboard and text events by widget focus
        before invoking this handler.
        """
        ...
    
    def on_text_motion(self, motion, select=...): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_text_motion` event.

        Caret keyboard handlers assume the layout always has keyboard focus.
        GUI toolkits should filter keyboard and text events by widget focus
        before invoking this handler.
        """
        ...
    
    def on_text_motion_select(self, motion): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_text_motion_select` event.

        Caret keyboard handlers assume the layout always has keyboard focus.
        GUI toolkits should filter keyboard and text events by widget focus
        before invoking this handler.
        """
        ...
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_mouse_scroll` event.

        Mouse handlers do not check the bounds of the coordinates: GUI
        toolkits should filter events that do not intersect the layout
        before invoking this handler.

        The layout viewport is scrolled by `SCROLL_INCREMENT` pixels per
        "click".
        """
        ...
    
    def on_mouse_press(self, x, y, button, modifiers): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_mouse_press` event.

        Mouse handlers do not check the bounds of the coordinates: GUI
        toolkits should filter events that do not intersect the layout
        before invoking this handler.

        This handler keeps track of the number of mouse presses within
        a short span of time and uses this to reconstruct double- and
        triple-click events for selecting words and paragraphs.  This
        technique is not suitable when a GUI toolkit is in use, as the active
        widget must also be tracked.  Do not use this mouse handler if
        a GUI toolkit is being used.
        """
        ...
    
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_mouse_drag` event.

        Mouse handlers do not check the bounds of the coordinates: GUI
        toolkits should filter events that do not intersect the layout
        before invoking this handler.
        """
        ...
    
    def on_activate(self): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_activate` event.

        The caret is hidden when the window is not active.
        """
        ...
    
    def on_deactivate(self): # -> Literal[True]:
        """Handler for the `pyglet.window.Window.on_deactivate` event.

        The caret is hidden when the window is not active.
        """
        ...
    


