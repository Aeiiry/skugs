import dearpygui.dearpygui as dpg
from skombo import utils

global FONTS
FONTS = utils.Fonts()

dpg.create_context()

# Font registry
with dpg.font_registry():
    defaultfont = dpg.add_font(FONTS.AtkinsonHyperlegible.regular.__str__(), 16)  # type: ignore

# Window
with dpg.window(label="Example Window", width=500, height=500):
    dpg.add_button(label="Button")

    dpg.bind_font(defaultfont)

dpg.create_viewport(title="Custom Title", width=1920, height=1080)

dpg.show_item_registry()
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
