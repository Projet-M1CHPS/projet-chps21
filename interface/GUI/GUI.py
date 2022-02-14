import sys

from PyQt5.QtCore import QFile, Qt
from PyQt5.QtDesigner import QFormBuilder
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget

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

app = QApplication(sys.argv)
file = QFile("./GUI.ui")
file.open(QFile.ReadOnly)
widget: QWidget = QFormBuilder().load(file)

tabWidget = widget.findChild(QTabWidget, "tabWidget")
if not tabWidget:
	exit(1)
tabs = tabWidget.children()[0].findChildren(QWidget, options=Qt.FindDirectChildrenOnly)
for each in tabs:
	print(each.objectName())

widget.show()
sys.exit(app.exec())
