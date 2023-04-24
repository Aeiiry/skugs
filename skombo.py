from tkinter import font
from tracemalloc import start
from PyQt5 import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import math
import os
import random
import sys


def createWindow(title) -> QMainWindow:
    window: QMainWindow = QMainWindow()
    window.setWindowTitle(title)

    return window

# Import font from .ttf file


def importFont(fontPath, size=20) -> QFont:
    font_id: int = QFontDatabase.addApplicationFont(fontPath)
    importFont: str = QFontDatabase.applicationFontFamilies(font_id)[0]
    font: QFont = QFont(importFont, size)
    return font


def createIconButton(iconPath) -> QPushButton:
    # create a button with an icon
    button: QPushButton = QPushButton()
    button.setIcon(QIcon(iconPath))
    return button


def createPushButton(text) -> QPushButton:
    # create a button with the text
    button: QPushButton = QPushButton(text)
    # set the font of the button to the skugfont

    return button


def createWindowLayout(window: QMainWindow) -> QGridLayout:
    # create a grid layout
    grid: QGridLayout = QGridLayout()
    # set the grid layout to the window
    window.setLayout(grid)
    return grid

# function to import a stylesheet


def importStyleSheet(stylePath) -> str:
    # check if the stylesheet exists
    if os.path.exists(stylePath):
        # open the stylesheet
        with open(stylePath, "r") as file:
            # read the stylesheet
            style: str = file.read()
            # return the stylesheet
            return style
    else:
        # stylesheet does not exist, so return an empty string
        return ""


def setRandomWindowIcon(window: QMainWindow) -> str:

    iconList: list[str] = os.listdir("data/program_icons")

    if len(iconList) == 0:

        window.setWindowIcon(QIcon("data/program_icons/default.png"))

    else:

        icon: str = iconList[math.floor(random.random() * len(iconList))]

        window.setWindowIcon(QIcon("data/program_icons/" + icon))
    return icon


# main function
def main() -> None:
    # Initialise the application
    skomboApp: QApplication = QApplication(sys.argv)
    # Create a new window
    mainWindow: QMainWindow = createWindow("skombo")
    # Import a custom font from the data/fonts folder
    skugfont: QFont = importFont(
        "data/fonts/setznick-nf/SelznickRemixNF.ttf", 20)
    # Set a random icon for the window and store the icon name in a variable
    randomIcon: str = setRandomWindowIcon(mainWindow)
    # Import a custom stylesheet from the data folder
    style: str = importStyleSheet("data/style_sheet.qss")
    # Set the stylesheet for the application
    skomboApp.setStyleSheet(style)
    # Set the font for the application
    skomboApp.setFont(skugfont)
    # Resize the window
    mainWindow.resize(1200, 1000)
    # Import the alignment flag from the Qt library for easier reference
    AlignmentFlag: Qt.AlignmentFlag = Qt.AlignmentFlag

    # create a menu bar
    menuBar: QMenuBar = QMenuBar()
    # add the menu bar to the window
    mainWindow.setMenuBar(menuBar)
    # create a tool bar
    toolBar: QToolBar = QToolBar()
    # add the tool bar to the window
    mainWindow.addToolBar(toolBar)
    # create a dock widget
    dockWidget: QDockWidget = QDockWidget()
    # add the dock widget to the window
    mainWindow.addDockWidget(Qt.LeftDockWidgetArea, dockWidget)
    # create a central widget
    centralWidget: QWidget = QWidget()
    # add the central widget to the window
    mainWindow.setCentralWidget(centralWidget)
    # create a status bar
    statusBar: QStatusBar = QStatusBar()
    # add the status bar to the window
    mainWindow.setStatusBar(statusBar)

    mainWindow.show()
    sys.exit(skomboApp.exec_())


pass


if __name__ == "__main__":
    main()
