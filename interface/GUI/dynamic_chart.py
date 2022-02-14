import _thread
import random
import sys
import time

import PyQt5.QtChart
import PyQt5.QtGui
from PyQt5.QtChart import QChartView
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout


def thread_fct(series: PyQt5.QtChart.QSplineSeries, chart_view: PyQt5.QtChart.QChartView, chart: PyQt5.QtChart.QChart):
    chart_view.setCacheMode(PyQt5.QtWidgets.QGraphicsView.CacheBackground)
    chart.setAnimationOptions(PyQt5.QtChart.QChart.SeriesAnimations)

    for i in range(90):
        time.sleep(.1)
        rand_value = random.random()
        series.append(len(series), rand_value)
        print(f"append {rand_value}")


def exportChartImage(filename, chart_view: PyQt5.QtChart.QChartView):
    pixel_map = chart_view.grab()
    gl_child: PyQt5.QtWidgets.QOpenGLWidget = chart_view.findChild(PyQt5.QtWidgets.QOpenGLWidget)
    if gl_child:
        painter = PyQt5.QtGui.QPainter(pixel_map)
        d = gl_child.mapToGlobal(PyQt5.QtCore.QPoint()) - chart_view.mapToGlobal(PyQt5.QtCore.QPoint())
        painter.setCompositionMode(PyQt5.QtGui.QPainter.CompositionMode_SourceAtop)
        painter.drawImage(d, gl_child.grabFramebuffer())
        painter.end()
    pixel_map.save(filename, "PNG")


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.setGeometry(150, 150, round(1920 * .7), round(1080 * .7))
        self.setWindowTitle("Creating LineChart")
        self.setWindowIcon(QIcon("python.png"))

        series = PyQt5.QtChart.QLineSeries()
        series.setName("test")
        series.setUseOpenGL(True)
        series.append(0, 1)
        for i in range(9):
            series.append(i, random.random())

        chart = PyQt5.QtChart.QChart()
        chart.addSeries(series)
        chart.setAnimationOptions(PyQt5.QtChart.QChart.SeriesAnimations)
        chart.setTitle("Line Chart Example")
        chart.setTheme(PyQt5.QtChart.QChart.ChartThemeDark)
        chart.createDefaultAxes()
        chart.axes(PyQt5.QtCore.Qt.Vertical)[0].setRange(-.05, 1.05)
        chart.axes(PyQt5.QtCore.Qt.Horizontal)[0].setRange(-1, 101)

        chart_view: QChartView = PyQt5.QtChart.QChartView(chart)
        chart_view.setRenderHint(PyQt5.QtGui.QPainter.Antialiasing)

        vbox = QVBoxLayout()
        vbox.addWidget(chart_view)
        self.setLayout(vbox)

        _thread.start_new_thread(thread_fct, (series, chart_view, chart,))


App = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(App.exec())
