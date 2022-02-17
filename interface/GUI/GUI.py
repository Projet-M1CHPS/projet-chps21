import sys

from PyQt5.QtCore import QFile
from PyQt5.QtDesigner import QFormBuilder
from PyQt5.QtWidgets import QApplication, QWidget


#
# class AppWindow(QWidget):
# 	def __init__(self):
# 		super().__init__()
# 		self.setGeometry(150, 150, round(1920 * .7), round(1080 * .7))
# 		self.setWindowTitle("Creating LineChart")
# 		self.setWindowIcon(QIcon("python.png"))
# 		builder = QFormBuilder()
# 		file = QFile("./GUI.ui")
# 		file.open(QFile.ReadOnly)
# 		widget = builder.load(file)
# 		file.close()
# 		layout = QVBoxLayout()
# 		layout.addWidget(widget)
# 		self.setLayout(layout)
#

# App = QApplication(sys.argv)
# window = AppWindow()
# window.show()
# sys.exit(App.exec())


def start():
    app = QApplication(sys.argv)
    file = QFile("GUI.ui")
    file.open(QFile.ReadOnly)
    widget: QWidget = QFormBuilder().load(file)
    if widget is None:
        print(".ui file can not be found.", file=sys.stderr)
        exit(1)

    # tab_widget = widget.findChild(QTabWidget, "tab_widget")
    # if not tab_widget:
    # 	print("no tab widegt")
    # 	exit(1)
    # tabs = tab_widget.children()[0].findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
    # for each in tabs:
    # 	print(each.objectName())

    widget.show()
    app.exec()


if __name__ == '__main__':
    start()
