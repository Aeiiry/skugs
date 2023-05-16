"""Gui"""
import os
from PyQt6.QtWidgets import (
    QPushButton as PushButton,
    QApplication as App,
    QWidget as Widget,
    QDialog as Dialog,
    QDialogButtonBox as DialogButtonBox,
    QFormLayout as FormLayout,
    QGridLayout as GridLayout,
    QLineEdit as LineEdit,
    QVBoxLayout as VBoxLayout,
    QMainWindow as MainWindow,
)
from PyQt6.QtCore import Qt
import skombo
from skombo.combo_calc import parse_combos_from_csv

ALINGN_LEFT = Qt.AlignmentFlag.AlignLeft
ALINGN_RIGHT = Qt.AlignmentFlag.AlignRight
ALINGN_CENTER = Qt.AlignmentFlag.AlignCenter
ALINGN_JUSTIFY = Qt.AlignmentFlag.AlignJustify


TEMP_BUTTON_SIZE = 100

INITIAL_WINDOW_HEIGHT = 480
INITIAL_WINDOW_WIDTH = 640


class SkomboWindow(MainWindow):
    def __init__(self):
        super().__init__(parent=None)

        self.setWindowTitle("Skug")

        self.resize(INITIAL_WINDOW_HEIGHT, INITIAL_WINDOW_WIDTH)

        self.general_layout = VBoxLayout()

        central_widget = Widget(self)
        central_widget.setLayout(self.general_layout)
        self.setCentralWidget(central_widget)
        self._create_display()
        self._create_buttons()

    def _create_display(self):
        self.display = LineEdit()
        self.display.setReadOnly(True)
        self.general_layout.addWidget(self.display)

    def _create_buttons(self):
        self.buttons = {}
        buttons_layout = GridLayout()
        button_texts = ["from csv", "just run the test"]

        for text in button_texts:
            self.buttons[text] = PushButton(text)
            self.buttons[text].setFixedSize(TEMP_BUTTON_SIZE, TEMP_BUTTON_SIZE)
            self.buttons[text].clicked.connect(
                parse_combos_from_csv(
                    os.path.join(
                        skombo.ABS_PATH,
                        (skombo.CHARACTERS["AN"].lower() + skombo.TEST_COMBOS_SUFFIX),
                    ),
                    calc_damage=True,
                )
            )
            buttons_layout.addWidget(self.buttons[text])

        self.general_layout.addLayout(buttons_layout)


def main():
    skombo_app = App([])
    window = SkomboWindow()
    window.show()
    skombo_app.exec()


if __name__ == "__main__":
    main()
