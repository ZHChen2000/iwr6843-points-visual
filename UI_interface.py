# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GestureRecognize.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QComboBox
import sys
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph.opengl import GLViewWidget
import globalvar as gl
import pyqtgraph as pg
import os
import matplotlib
import matplotlib.cm
import time
import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
gl._init()
gl.set_value('usr_gesture', False)


radarRFconfigpathfile = 'config'
modelpathfile = 'save_model'


# ----新的combox 单机框任何地方都有响应 ---- #
class ClickedComboBox(QComboBox):
    arrowClicked = pyqtSignal()

    def showPopup(self):  # 重写showPopup函数
        super(ClickedComboBox, self).showPopup()
        self.arrowClicked.emit()

# 最小化窗口
class Qt_pet(QWidget):

    def __init__(self,MainWindow):
        super(Qt_pet, self).__init__()
        self.MainWindow = MainWindow
        self.dis_file = "gesture_icons/"
        self.windowinit()

        self.pos_first = self.pos()
        # self.timer.timeout.connect(self.img_update)

    def img_update(self,img_path):
        self.img_path = img_path
        self.qpixmap = QPixmap(self.img_path).scaled(256, 256)
        self.lab.setPixmap(self.qpixmap)

    def windowinit(self):
        self.x = 800
        self.y = 600
        self.setGeometry(self.x, self.y, 256, 256)
        self.img_path = 'gesture_icons/7.jpg'
        self.lab = QLabel(self)
        self.qpixmap = QPixmap(self.img_path).scaled(256, 256)
        self.lab.setPixmap(self.qpixmap)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # self.show()

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.pos_first = QMouseEvent.globalPos() - self.pos()
            QMouseEvent.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))
        if QMouseEvent.button() == Qt.RightButton:
            self.MainWindow.show()
            self.hide()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton:
            self.move(QMouseEvent.globalPos() - self.pos_first)
            # print(self.pos())
            self.x, self.y = self.pos().x, self.pos().y
            QMouseEvent.accept()


# MainWindow主窗口
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1200)
        pg.setConfigOption('background', '#f0f0f0')
        pg.setConfigOption('foreground', 'd')         
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        # centralwidget主内容容器
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        # tabwidget选项卡容器，顶部两个选项卡
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        # splitter横向分割左右区域，左侧图像显示，右侧配置面板
        self.splitter = QtWidgets.QSplitter(self.tab)
        self.splitter.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setAutoFillBackground(False)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        # groupbox_11为右侧所有控件的顶层容器
        self.groupBox_11 = QtWidgets.QGroupBox(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(15)
        sizePolicy.setHeightForWidth(self.groupBox_11.sizePolicy().hasHeightForWidth())
        self.groupBox_11.setSizePolicy(sizePolicy)
        self.groupBox_11.setMinimumSize(QtCore.QSize(0, 0))
        self.groupBox_11.setMaximumSize(QtCore.QSize(2400, 8000))
        self.groupBox_11.setObjectName("groupBox_11")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_11)
        self.verticalLayout_4.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_11)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_15 = QtWidgets.QLabel(self.groupBox_2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_3.addWidget(self.label_15, 3, 0, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox_2)
        self.label_16.setObjectName("label_16")
        self.gridLayout_3.addWidget(self.label_16, 3, 1, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.groupBox_2)
        self.label_36.setObjectName("label_36")
        self.gridLayout_3.addWidget(self.label_36, 4, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 1, 1, 1, 1)
        self.label_43 = QtWidgets.QLabel(self.groupBox_2)
        self.label_43.setObjectName("label_43")
        self.gridLayout_3.addWidget(self.label_43, 6, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.groupBox_2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 1, 0, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.groupBox_2)
        self.label_35.setObjectName("label_35")
        self.gridLayout_3.addWidget(self.label_35, 2, 1, 1, 1)
        self.comboBox_7 = ClickedComboBox(self.groupBox_2)
        self.comboBox_7.setObjectName("comboBox_7")
        self.gridLayout_3.addWidget(self.comboBox_7, 0, 1, 1, 1)
        self.label_45 = QtWidgets.QLabel(self.groupBox_2)
        self.label_45.setAlignment(QtCore.Qt.AlignCenter)
        self.label_45.setObjectName("label_45")
        self.gridLayout_3.addWidget(self.label_45, 5, 0, 1, 2)
        self.comboBox_9 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_9.setEnabled(False)
        self.comboBox_9.setEditable(False)
        self.comboBox_9.setObjectName("comboBox_9")
        self.gridLayout_3.addWidget(self.comboBox_9, 7, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 0, 0, 1, 1)
        self.label_44 = QtWidgets.QLabel(self.groupBox_2)
        self.label_44.setEnabled(False)
        self.label_44.setObjectName("label_44")
        self.gridLayout_3.addWidget(self.label_44, 7, 0, 1, 1)
        self.label_46 = QtWidgets.QLabel(self.groupBox_2)
        self.label_46.setObjectName("label_46")
        self.gridLayout_3.addWidget(self.label_46, 8, 0, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.groupBox_2)
        self.label_17.setObjectName("label_17")
        self.gridLayout_3.addWidget(self.label_17, 2, 0, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.groupBox_2)
        self.label_37.setObjectName("label_37")
        self.gridLayout_3.addWidget(self.label_37, 4, 1, 1, 1)
        self.comboBox_8 = ClickedComboBox(self.groupBox_2)
        self.comboBox_8.setObjectName("comboBox_8")
        self.gridLayout_3.addWidget(self.comboBox_8, 6, 1, 1, 1)
        self.label_46 = QtWidgets.QLabel(self.groupBox_2)
        self.label_46.setObjectName("label_46")
        self.gridLayout_3.addWidget(self.label_46, 7, 0, 1, 1)
        self.comboBox_10 = ClickedComboBox(self.groupBox_2)
        self.comboBox_10.setObjectName("comboBox_10")
        self.gridLayout_3.addWidget(self.comboBox_10, 7, 1, 1, 1)
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_12.setObjectName("pushButton_12")
        self.gridLayout_3.addWidget(self.pushButton_11, 9, 1, 1, 1)
        self.gridLayout_3.addWidget(self.pushButton_12, 9, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.groupBox_7.sizePolicy().hasHeightForWidth())
        self.groupBox_7.setSizePolicy(sizePolicy)
        self.groupBox_7.setObjectName("groupBox_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_7)
        self.verticalLayout_7.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_5 = QtWidgets.QLabel(self.groupBox_7)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout.addWidget(self.label_5)
        self.comboBox = ClickedComboBox(self.groupBox_7)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout.addWidget(self.comboBox)
        self.verticalLayout_7.addLayout(self.horizontalLayout)
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_7)
        self.groupBox_3.setFlat(True)
        self.groupBox_3.setCheckable(True)
        self.groupBox_3.setChecked(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_5.setContentsMargins(2, 6, 2, 6)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.comboBox_2 = ClickedComboBox(self.groupBox_3)
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_3.addWidget(self.comboBox_2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.pushButton_15 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_15.setCheckable(True)
        self.verticalLayout_5.addWidget(self.pushButton_15)
        self.verticalLayout_7.addWidget(self.groupBox_3)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_7)
        self.groupBox_4.setFlat(True)
        self.groupBox_4.setCheckable(True)
        self.groupBox_4.setChecked(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setContentsMargins(2, 6, 2, 6)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.groupBox_4)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.horizontalLayout_5.addWidget(self.lineEdit_6)
        self.comboBox_3 = QtWidgets.QComboBox(self.groupBox_4)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.horizontalLayout_5.addWidget(self.comboBox_3)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.pushButton = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton.setCheckable(True)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_6.addWidget(self.pushButton)
        self.verticalLayout_7.addWidget(self.groupBox_4)
        self.verticalLayout_4.addWidget(self.groupBox_7)
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_11)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.textEdit.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.textEdit.setTabStopWidth(80)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_2.addWidget(self.textEdit)
        self.verticalLayout_4.addWidget(self.groupBox)
        # groupbox_9为左侧六图显示区容器
        self.groupBox_9 = QtWidgets.QGroupBox(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(15)
        sizePolicy.setHeightForWidth(self.groupBox_9.sizePolicy().hasHeightForWidth())
        self.groupBox_9.setSizePolicy(sizePolicy)
        self.groupBox_9.setObjectName("groupBox_9")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_9)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setHorizontalSpacing(7)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 2, 1, 1)
        self.graphicsView_4 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_4.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsView_4.sizePolicy().hasHeightForWidth())
        self.graphicsView_4.setSizePolicy(sizePolicy)
        self.graphicsView_4.setMinimumSize(QtCore.QSize(400, 400))
        self.graphicsView_4.setMaximumSize(QtCore.QSize(400, 400))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.gridLayout.addWidget(self.graphicsView_4, 3, 1, 1, 1)
        self.graphicsView_2 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_2.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsView_2.sizePolicy().hasHeightForWidth())
        self.graphicsView_2.setSizePolicy(sizePolicy)
        self.graphicsView_2.setMinimumSize(QtCore.QSize(400, 400))
        self.graphicsView_2.setMaximumSize(QtCore.QSize(400, 400))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout.addWidget(self.graphicsView_2, 1, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 2, 1, 1, 1)
        self.graphicsView = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)

        # ZHChen 修改了时间-距离图的控件大小，原尺寸为255，255 时间：2025-05-20
        self.graphicsView.setMinimumSize(QtCore.QSize(400, 400))
        self.graphicsView.setMaximumSize(QtCore.QSize(400, 400))
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        self.graphicsView_3 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_3.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsView_3.sizePolicy().hasHeightForWidth())
        self.graphicsView_3.setSizePolicy(sizePolicy)
        self.graphicsView_3.setMinimumSize(QtCore.QSize(400, 400))
        self.graphicsView_3.setMaximumSize(QtCore.QSize(400, 400))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.gridLayout.addWidget(self.graphicsView_3, 1, 2, 1, 1)
        self.graphicsView_5 = QtWidgets.QLabel(self.groupBox_9)
        self.graphicsView_5.setStyleSheet('border-width: 2px;border-style: solid;border-color: rgb(255, 170, 0);background-color: rgb(180,180,180);')
        self.graphicsView_5.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsView_5.sizePolicy().hasHeightForWidth())
        self.graphicsView_5.setSizePolicy(sizePolicy)
        self.graphicsView_5.setMinimumSize(QtCore.QSize(400, 400))
        self.graphicsView_5.setMaximumSize(QtCore.QSize(400, 400))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.gridLayout.addWidget(self.graphicsView_5, 3, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 2, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 2, 1, 1)
        self.graphicsView_6 = GraphicsLayoutWidget(self.groupBox_9)
        self.graphicsView_6.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.graphicsView_6.sizePolicy().hasHeightForWidth())
        self.graphicsView_6.setSizePolicy(sizePolicy)
        self.graphicsView_6.setMinimumSize(QtCore.QSize(400, 400))
        self.graphicsView_6.setMaximumSize(QtCore.QSize(400, 400))
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.gridLayout.addWidget(self.graphicsView_6, 3, 0, 1, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 4)
        self.gridLayout.setRowStretch(2, 1)
        self.gridLayout.setRowStretch(3, 4)
        self.verticalLayout_3.addWidget(self.splitter)

        # 比例控制 splitter 内控件的比例
        self.splitter.addWidget(self.groupBox_9)#'你的第一个子 widget'
        self.splitter.addWidget(self.groupBox_11)#'你的第二个子 widget'
        self.splitter.setSizes([960,320])#直接写入数字列表

        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_4.setPixmap(QtGui.QPixmap("visualization/depict.jpg"))
        self.label_4.setObjectName("label_4")
        self.verticalLayout_8.addWidget(self.label_4)
        self.tabWidget.addTab(self.tab_2, "")
        # 添加点云显示选项卡
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        
        # 创建水平布局，左侧显示，右侧配置
        self.horizontalLayout_pointcloud = QtWidgets.QHBoxLayout()
        self.horizontalLayout_pointcloud.setObjectName("horizontalLayout_pointcloud")
        
        # 左侧：3D显示区域
        self.verticalLayout_pointcloud_display = QtWidgets.QVBoxLayout()
        self.verticalLayout_pointcloud_display.setObjectName("verticalLayout_pointcloud_display")
        # 点云3D显示区域（使用GLViewWidget用于3D显示）
        self.graphicsView_pointcloud = GLViewWidget(self.tab_3)
        self.graphicsView_pointcloud.setObjectName("graphicsView_pointcloud")
        self.graphicsView_pointcloud.setMinimumHeight(400)  # 设置最小高度，确保可见
        # 设置stretch factor为10，让3D显示区域占据大部分空间
        self.verticalLayout_pointcloud_display.addWidget(self.graphicsView_pointcloud, 10)
        # 点云信息显示标签
        self.label_pointcloud_info = QtWidgets.QLabel(self.tab_3)
        self.label_pointcloud_info.setObjectName("label_pointcloud_info")
        self.label_pointcloud_info.setText("点云数量: 0")
        self.label_pointcloud_info.setMaximumHeight(30)  # 限制标签高度
        # 设置stretch factor为0，不占用额外空间
        self.verticalLayout_pointcloud_display.addWidget(self.label_pointcloud_info, 0)
        
        # 坐标信息显示面板（已隐藏）
        # self.groupBox_coordinate_info = QtWidgets.QGroupBox(self.tab_3)
        # self.groupBox_coordinate_info.setObjectName("groupBox_coordinate_info")
        # self.groupBox_coordinate_info.setTitle("坐标与方位信息")
        # self.groupBox_coordinate_info.setMaximumHeight(120)
        # self.verticalLayout_coordinate_info = QtWidgets.QVBoxLayout(self.groupBox_coordinate_info)
        # self.verticalLayout_coordinate_info.setContentsMargins(5, 5, 5, 5)
        # self.verticalLayout_coordinate_info.setObjectName("verticalLayout_coordinate_info")
        
        # 坐标信息标签（已隐藏）
        # self.label_coordinate_info = QtWidgets.QLabel(self.groupBox_coordinate_info)
        # self.label_coordinate_info.setObjectName("label_coordinate_info")
        # self.label_coordinate_info.setText("等待点云数据...")
        # self.label_coordinate_info.setWordWrap(True)  # 允许换行
        # self.verticalLayout_coordinate_info.addWidget(self.label_coordinate_info)
        # self.verticalLayout_pointcloud_display.addWidget(self.groupBox_coordinate_info, 0)
        
        # 创建占位标签以避免引用错误
        self.label_coordinate_info = QtWidgets.QLabel()
        self.label_coordinate_info.setObjectName("label_coordinate_info")
        
        # 点云采集日志显示区域
        self.groupBox_pointcloud_log = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_pointcloud_log.setObjectName("groupBox_pointcloud_log")
        self.groupBox_pointcloud_log.setTitle("点云采集日志")
        self.verticalLayout_pointcloud_log = QtWidgets.QVBoxLayout(self.groupBox_pointcloud_log)
        self.verticalLayout_pointcloud_log.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout_pointcloud_log.setObjectName("verticalLayout_pointcloud_log")
        self.textEdit_pointcloud_log = QtWidgets.QTextEdit(self.groupBox_pointcloud_log)
        self.textEdit_pointcloud_log.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.textEdit_pointcloud_log.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.textEdit_pointcloud_log.setReadOnly(True)  # 只读
        self.textEdit_pointcloud_log.setMaximumHeight(200)  # 限制高度
        self.textEdit_pointcloud_log.setMinimumHeight(150)  # 设置最小高度
        self.textEdit_pointcloud_log.setObjectName("textEdit_pointcloud_log")
        self.verticalLayout_pointcloud_log.addWidget(self.textEdit_pointcloud_log)
        # 设置stretch factor为0，不占用额外空间（已经有最大高度限制）
        self.verticalLayout_pointcloud_display.addWidget(self.groupBox_pointcloud_log, 0)
        
        # 右侧：配置面板
        self.groupBox_pointcloud_config = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_pointcloud_config.setObjectName("groupBox_pointcloud_config")
        self.groupBox_pointcloud_config.setMaximumWidth(250)
        self.verticalLayout_pointcloud_config = QtWidgets.QVBoxLayout(self.groupBox_pointcloud_config)
        self.verticalLayout_pointcloud_config.setObjectName("verticalLayout_pointcloud_config")
        
        # 点云数量限制
        self.label_max_points = QtWidgets.QLabel(self.groupBox_pointcloud_config)
        self.label_max_points.setObjectName("label_max_points")
        self.label_max_points.setText("最大点云数量:")
        self.verticalLayout_pointcloud_config.addWidget(self.label_max_points)
        self.spinBox_max_points = QtWidgets.QSpinBox(self.groupBox_pointcloud_config)
        self.spinBox_max_points.setObjectName("spinBox_max_points")
        self.spinBox_max_points.setMinimum(10)
        self.spinBox_max_points.setMaximum(10000)
        self.spinBox_max_points.setValue(1000)
        self.verticalLayout_pointcloud_config.addWidget(self.spinBox_max_points)
        
        # 刷新速率
        self.label_refresh_rate = QtWidgets.QLabel(self.groupBox_pointcloud_config)
        self.label_refresh_rate.setObjectName("label_refresh_rate")
        self.label_refresh_rate.setText("刷新速率 (ms):")
        self.verticalLayout_pointcloud_config.addWidget(self.label_refresh_rate)
        self.spinBox_refresh_rate = QtWidgets.QSpinBox(self.groupBox_pointcloud_config)
        self.spinBox_refresh_rate.setObjectName("spinBox_refresh_rate")
        self.spinBox_refresh_rate.setMinimum(1)
        self.spinBox_refresh_rate.setMaximum(1000)
        self.spinBox_refresh_rate.setValue(20)
        self.verticalLayout_pointcloud_config.addWidget(self.spinBox_refresh_rate)
        
        # 阈值比例
        self.label_threshold = QtWidgets.QLabel(self.groupBox_pointcloud_config)
        self.label_threshold.setObjectName("label_threshold")
        self.label_threshold.setText("检测阈值比例:")
        self.verticalLayout_pointcloud_config.addWidget(self.label_threshold)
        self.doubleSpinBox_threshold = QtWidgets.QDoubleSpinBox(self.groupBox_pointcloud_config)
        self.doubleSpinBox_threshold.setObjectName("doubleSpinBox_threshold")
        self.doubleSpinBox_threshold.setMinimum(0.05)  # 降低最小值，允许更低的阈值
        self.doubleSpinBox_threshold.setMaximum(1.0)
        self.doubleSpinBox_threshold.setSingleStep(0.05)  # 减小步长
        self.doubleSpinBox_threshold.setValue(0.99)  # 默认值0.99
        self.verticalLayout_pointcloud_config.addWidget(self.doubleSpinBox_threshold)
        
        # 添加测试点云按钮
        self.pushButton_test_pointcloud = QtWidgets.QPushButton(self.groupBox_pointcloud_config)
        self.pushButton_test_pointcloud.setObjectName("pushButton_test_pointcloud")
        self.pushButton_test_pointcloud.setText("测试点云")
        self.verticalLayout_pointcloud_config.addWidget(self.pushButton_test_pointcloud)
        
        # 显示网格
        self.checkBox_show_grid = QtWidgets.QCheckBox(self.groupBox_pointcloud_config)
        self.checkBox_show_grid.setObjectName("checkBox_show_grid")
        self.checkBox_show_grid.setText("显示网格")
        self.checkBox_show_grid.setChecked(True)
        self.verticalLayout_pointcloud_config.addWidget(self.checkBox_show_grid)
        
        # 显示坐标系
        self.checkBox_show_axes = QtWidgets.QCheckBox(self.groupBox_pointcloud_config)
        self.checkBox_show_axes.setObjectName("checkBox_show_axes")
        self.checkBox_show_axes.setText("显示坐标系")
        self.checkBox_show_axes.setChecked(True)
        self.verticalLayout_pointcloud_config.addWidget(self.checkBox_show_axes)
        
        # 添加分隔线
        separator_line = QtWidgets.QFrame(self.groupBox_pointcloud_config)
        separator_line.setFrameShape(QtWidgets.QFrame.HLine)
        separator_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_pointcloud_config.addWidget(separator_line)
        
        # 聚类功能开关
        self.checkBox_enable_clustering = QtWidgets.QCheckBox(self.groupBox_pointcloud_config)
        self.checkBox_enable_clustering.setObjectName("checkBox_enable_clustering")
        self.checkBox_enable_clustering.setText("显示聚类后的点云")
        self.checkBox_enable_clustering.setChecked(True)
        self.verticalLayout_pointcloud_config.addWidget(self.checkBox_enable_clustering)
        
        # 聚类距离阈值
        self.label_cluster_eps = QtWidgets.QLabel(self.groupBox_pointcloud_config)
        self.label_cluster_eps.setObjectName("label_cluster_eps")
        self.label_cluster_eps.setText("聚类距离阈值 (m):")
        self.verticalLayout_pointcloud_config.addWidget(self.label_cluster_eps)
        self.doubleSpinBox_cluster_eps = QtWidgets.QDoubleSpinBox(self.groupBox_pointcloud_config)
        self.doubleSpinBox_cluster_eps.setObjectName("doubleSpinBox_cluster_eps")
        self.doubleSpinBox_cluster_eps.setMinimum(0.05)
        self.doubleSpinBox_cluster_eps.setMaximum(2.0)
        self.doubleSpinBox_cluster_eps.setSingleStep(0.05)
        self.doubleSpinBox_cluster_eps.setValue(0.05)
        self.verticalLayout_pointcloud_config.addWidget(self.doubleSpinBox_cluster_eps)
        
        # 最小聚类点数
        self.label_cluster_min_samples = QtWidgets.QLabel(self.groupBox_pointcloud_config)
        self.label_cluster_min_samples.setObjectName("label_cluster_min_samples")
        self.label_cluster_min_samples.setText("最小聚类点数:")
        self.verticalLayout_pointcloud_config.addWidget(self.label_cluster_min_samples)
        self.spinBox_cluster_min_samples = QtWidgets.QSpinBox(self.groupBox_pointcloud_config)
        self.spinBox_cluster_min_samples.setObjectName("spinBox_cluster_min_samples")
        self.spinBox_cluster_min_samples.setMinimum(2)
        self.spinBox_cluster_min_samples.setMaximum(20)
        self.spinBox_cluster_min_samples.setValue(3)
        self.verticalLayout_pointcloud_config.addWidget(self.spinBox_cluster_min_samples)
        
        # 添加弹性空间
        self.verticalLayout_pointcloud_config.addStretch()
        
        # 组合布局 - 调整比例，让可视化部分更大（从3:1改为6:1）
        self.horizontalLayout_pointcloud.addLayout(self.verticalLayout_pointcloud_display, 6)
        self.horizontalLayout_pointcloud.addWidget(self.groupBox_pointcloud_config, 1)
        self.verticalLayout_9.addLayout(self.horizontalLayout_pointcloud)
        
        self.tabWidget.addTab(self.tab_3, "")
        
        # 添加轨迹显示选项卡
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_trajectory = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_trajectory.setObjectName("verticalLayout_trajectory")
        
        # 创建水平布局，左侧显示，右侧配置
        self.horizontalLayout_trajectory = QtWidgets.QHBoxLayout()
        self.horizontalLayout_trajectory.setObjectName("horizontalLayout_trajectory")
        
        # 左侧：轨迹显示区域
        self.verticalLayout_trajectory_display = QtWidgets.QVBoxLayout()
        self.verticalLayout_trajectory_display.setObjectName("verticalLayout_trajectory_display")
        
        # 轨迹2D显示区域
        self.graphicsView_trajectory = GraphicsLayoutWidget(self.tab_4)
        self.graphicsView_trajectory.setObjectName("graphicsView_trajectory")
        self.graphicsView_trajectory.setMinimumHeight(600)
        self.verticalLayout_trajectory_display.addWidget(self.graphicsView_trajectory, 10)
        
        # 轨迹信息显示标签
        self.label_trajectory_info = QtWidgets.QLabel(self.tab_4)
        self.label_trajectory_info.setObjectName("label_trajectory_info")
        self.label_trajectory_info.setText("轨迹信息: 等待数据...")
        self.label_trajectory_info.setMaximumHeight(30)
        self.verticalLayout_trajectory_display.addWidget(self.label_trajectory_info, 0)
        
        # 右侧：配置面板
        self.groupBox_trajectory_config = QtWidgets.QGroupBox(self.tab_4)
        self.groupBox_trajectory_config.setObjectName("groupBox_trajectory_config")
        self.groupBox_trajectory_config.setTitle("轨迹配置")
        self.groupBox_trajectory_config.setMaximumWidth(250)
        self.verticalLayout_trajectory_config = QtWidgets.QVBoxLayout(self.groupBox_trajectory_config)
        self.verticalLayout_trajectory_config.setObjectName("verticalLayout_trajectory_config")
        
        # 轨迹历史长度
        self.label_trajectory_history = QtWidgets.QLabel(self.groupBox_trajectory_config)
        self.label_trajectory_history.setObjectName("label_trajectory_history")
        self.label_trajectory_history.setText("轨迹历史长度:")
        self.verticalLayout_trajectory_config.addWidget(self.label_trajectory_history)
        self.spinBox_trajectory_history = QtWidgets.QSpinBox(self.groupBox_trajectory_config)
        self.spinBox_trajectory_history.setObjectName("spinBox_trajectory_history")
        self.spinBox_trajectory_history.setMinimum(10)
        self.spinBox_trajectory_history.setMaximum(1000)
        self.spinBox_trajectory_history.setValue(15)
        self.verticalLayout_trajectory_config.addWidget(self.spinBox_trajectory_history)
        
        # 轨迹点大小
        self.label_trajectory_point_size = QtWidgets.QLabel(self.groupBox_trajectory_config)
        self.label_trajectory_point_size.setObjectName("label_trajectory_point_size")
        self.label_trajectory_point_size.setText("轨迹点大小:")
        self.verticalLayout_trajectory_config.addWidget(self.label_trajectory_point_size)
        self.spinBox_trajectory_point_size = QtWidgets.QSpinBox(self.groupBox_trajectory_config)
        self.spinBox_trajectory_point_size.setObjectName("spinBox_trajectory_point_size")
        self.spinBox_trajectory_point_size.setMinimum(1)
        self.spinBox_trajectory_point_size.setMaximum(50)
        self.spinBox_trajectory_point_size.setValue(40)
        self.verticalLayout_trajectory_config.addWidget(self.spinBox_trajectory_point_size)
        
        # 显示坐标轴
        self.checkBox_trajectory_show_axes = QtWidgets.QCheckBox(self.groupBox_trajectory_config)
        self.checkBox_trajectory_show_axes.setObjectName("checkBox_trajectory_show_axes")
        self.checkBox_trajectory_show_axes.setText("显示坐标轴")
        self.checkBox_trajectory_show_axes.setChecked(True)
        self.verticalLayout_trajectory_config.addWidget(self.checkBox_trajectory_show_axes)
        
        # 显示网格
        self.checkBox_trajectory_show_grid = QtWidgets.QCheckBox(self.groupBox_trajectory_config)
        self.checkBox_trajectory_show_grid.setObjectName("checkBox_trajectory_show_grid")
        self.checkBox_trajectory_show_grid.setText("显示网格")
        self.checkBox_trajectory_show_grid.setChecked(True)
        self.verticalLayout_trajectory_config.addWidget(self.checkBox_trajectory_show_grid)
        
        # 待检测目标个数
        self.label_target_count = QtWidgets.QLabel(self.groupBox_trajectory_config)
        self.label_target_count.setObjectName("label_target_count")
        self.label_target_count.setText("待检测目标个数:")
        self.verticalLayout_trajectory_config.addWidget(self.label_target_count)
        self.spinBox_target_count = QtWidgets.QSpinBox(self.groupBox_trajectory_config)
        self.spinBox_target_count.setObjectName("spinBox_target_count")
        self.spinBox_target_count.setMinimum(1)
        self.spinBox_target_count.setMaximum(10)
        self.spinBox_target_count.setValue(1)
        self.verticalLayout_trajectory_config.addWidget(self.spinBox_target_count)
        
        # 清除轨迹按钮
        self.pushButton_clear_trajectory = QtWidgets.QPushButton(self.groupBox_trajectory_config)
        self.pushButton_clear_trajectory.setObjectName("pushButton_clear_trajectory")
        self.pushButton_clear_trajectory.setText("清除轨迹")
        self.verticalLayout_trajectory_config.addWidget(self.pushButton_clear_trajectory)
        
        # 添加弹性空间
        self.verticalLayout_trajectory_config.addStretch()
        
        # 组合布局
        self.horizontalLayout_trajectory.addLayout(self.verticalLayout_trajectory_display, 6)
        self.horizontalLayout_trajectory.addWidget(self.groupBox_trajectory_config, 1)
        self.verticalLayout_trajectory.addLayout(self.horizontalLayout_trajectory)
        
        self.tabWidget.addTab(self.tab_4, "")
        
        # 添加高度检测选项卡
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.verticalLayout_height = QtWidgets.QVBoxLayout(self.tab_5)
        self.verticalLayout_height.setObjectName("verticalLayout_height")
        
        # 创建水平布局，左侧显示，右侧配置
        self.horizontalLayout_height = QtWidgets.QHBoxLayout()
        self.horizontalLayout_height.setObjectName("horizontalLayout_height")
        
        # 左侧：高度显示区域
        self.verticalLayout_height_display = QtWidgets.QVBoxLayout()
        self.verticalLayout_height_display.setObjectName("verticalLayout_height_display")
        
        # 高度2D显示区域
        self.graphicsView_height = GraphicsLayoutWidget(self.tab_5)
        self.graphicsView_height.setObjectName("graphicsView_height")
        self.graphicsView_height.setMinimumHeight(600)
        self.verticalLayout_height_display.addWidget(self.graphicsView_height, 10)
        
        # 高度信息显示标签
        self.label_height_info = QtWidgets.QLabel(self.tab_5)
        self.label_height_info.setObjectName("label_height_info")
        self.label_height_info.setText("高度信息: 等待数据...")
        self.label_height_info.setMaximumHeight(30)
        self.verticalLayout_height_display.addWidget(self.label_height_info, 0)
        
        # 跌倒提示标签（大大的提示）
        self.label_fall_alert = QtWidgets.QLabel(self.tab_5)
        self.label_fall_alert.setObjectName("label_fall_alert")
        self.label_fall_alert.setText("")
        self.label_fall_alert.setAlignment(QtCore.Qt.AlignCenter)
        self.label_fall_alert.setStyleSheet("""
            QLabel {
                font-size: 72px;
                font-weight: bold;
                color: red;
                background-color: rgba(255, 255, 255, 200);
                border: 5px solid red;
                border-radius: 20px;
                padding: 20px;
            }
        """)
        self.label_fall_alert.hide()  # 默认隐藏
        self.verticalLayout_height_display.addWidget(self.label_fall_alert, 0)
        
        # 右侧：配置面板
        self.groupBox_height_config = QtWidgets.QGroupBox(self.tab_5)
        self.groupBox_height_config.setObjectName("groupBox_height_config")
        self.groupBox_height_config.setTitle("高度配置")
        self.groupBox_height_config.setMaximumWidth(250)
        self.verticalLayout_height_config = QtWidgets.QVBoxLayout(self.groupBox_height_config)
        self.verticalLayout_height_config.setObjectName("verticalLayout_height_config")
        
        # 高度历史长度
        self.label_height_history = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_height_history.setObjectName("label_height_history")
        self.label_height_history.setText("高度历史长度:")
        self.verticalLayout_height_config.addWidget(self.label_height_history)
        self.spinBox_height_history = QtWidgets.QSpinBox(self.groupBox_height_config)
        self.spinBox_height_history.setObjectName("spinBox_height_history")
        self.spinBox_height_history.setMinimum(10)
        self.spinBox_height_history.setMaximum(1000)
        self.spinBox_height_history.setValue(50)
        self.verticalLayout_height_config.addWidget(self.spinBox_height_history)
        
        # 高度点大小
        self.label_height_point_size = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_height_point_size.setObjectName("label_height_point_size")
        self.label_height_point_size.setText("高度点大小:")
        self.verticalLayout_height_config.addWidget(self.label_height_point_size)
        self.spinBox_height_point_size = QtWidgets.QSpinBox(self.groupBox_height_config)
        self.spinBox_height_point_size.setObjectName("spinBox_height_point_size")
        self.spinBox_height_point_size.setMinimum(1)
        self.spinBox_height_point_size.setMaximum(50)
        self.spinBox_height_point_size.setValue(40)
        self.verticalLayout_height_config.addWidget(self.spinBox_height_point_size)
        
        # 显示坐标轴
        self.checkBox_height_show_axes = QtWidgets.QCheckBox(self.groupBox_height_config)
        self.checkBox_height_show_axes.setObjectName("checkBox_height_show_axes")
        self.checkBox_height_show_axes.setText("显示坐标轴")
        self.checkBox_height_show_axes.setChecked(True)
        self.verticalLayout_height_config.addWidget(self.checkBox_height_show_axes)
        
        # 显示网格
        self.checkBox_height_show_grid = QtWidgets.QCheckBox(self.groupBox_height_config)
        self.checkBox_height_show_grid.setObjectName("checkBox_height_show_grid")
        self.checkBox_height_show_grid.setText("显示网格")
        self.checkBox_height_show_grid.setChecked(True)
        self.verticalLayout_height_config.addWidget(self.checkBox_height_show_grid)
        
        # 添加分隔线
        separator_line_height = QtWidgets.QFrame(self.groupBox_height_config)
        separator_line_height.setFrameShape(QtWidgets.QFrame.HLine)
        separator_line_height.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_height_config.addWidget(separator_line_height)
        
        # 跌倒检测配置
        self.label_fall_detection = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_fall_detection.setObjectName("label_fall_detection")
        self.label_fall_detection.setText("跌倒检测配置:")
        self.label_fall_detection.setStyleSheet("font-weight: bold;")
        self.verticalLayout_height_config.addWidget(self.label_fall_detection)
        
        # 跌倒检测开关
        self.checkBox_enable_fall_detection = QtWidgets.QCheckBox(self.groupBox_height_config)
        self.checkBox_enable_fall_detection.setObjectName("checkBox_enable_fall_detection")
        self.checkBox_enable_fall_detection.setText("启用跌倒检测")
        self.checkBox_enable_fall_detection.setChecked(True)  # 默认启用
        self.verticalLayout_height_config.addWidget(self.checkBox_enable_fall_detection)
        
        # 启用新策略跌倒检测
        self.checkBox_enable_new_fall_strategy = QtWidgets.QCheckBox(self.groupBox_height_config)
        self.checkBox_enable_new_fall_strategy.setObjectName("checkBox_enable_new_fall_strategy")
        self.checkBox_enable_new_fall_strategy.setText("启用新策略跌倒检测（多特征融合）")
        self.checkBox_enable_new_fall_strategy.setChecked(False)  # 默认关闭，用户可选择启用
        self.verticalLayout_height_config.addWidget(self.checkBox_enable_new_fall_strategy)
        
        # 新策略灵敏度等级选择
        self.label_new_strategy_sensitivity = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_new_strategy_sensitivity.setObjectName("label_new_strategy_sensitivity")
        self.label_new_strategy_sensitivity.setText("新策略灵敏度等级:")
        self.verticalLayout_height_config.addWidget(self.label_new_strategy_sensitivity)
        self.comboBox_new_strategy_sensitivity = QtWidgets.QComboBox(self.groupBox_height_config)
        self.comboBox_new_strategy_sensitivity.setObjectName("comboBox_new_strategy_sensitivity")
        self.comboBox_new_strategy_sensitivity.addItems(["灵敏", "中等", "不灵敏"])
        self.comboBox_new_strategy_sensitivity.setCurrentIndex(1)  # 默认选择"中等"
        self.verticalLayout_height_config.addWidget(self.comboBox_new_strategy_sensitivity)
        
        # 检测时间窗口（秒）
        self.label_fall_time_window = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_fall_time_window.setObjectName("label_fall_time_window")
        self.label_fall_time_window.setText("检测时间窗口 (秒):")
        self.verticalLayout_height_config.addWidget(self.label_fall_time_window)
        self.doubleSpinBox_fall_time_window = QtWidgets.QDoubleSpinBox(self.groupBox_height_config)
        self.doubleSpinBox_fall_time_window.setObjectName("doubleSpinBox_fall_time_window")
        self.doubleSpinBox_fall_time_window.setMinimum(0.1)
        self.doubleSpinBox_fall_time_window.setMaximum(10.0)
        self.doubleSpinBox_fall_time_window.setSingleStep(0.1)
        self.doubleSpinBox_fall_time_window.setValue(2.0)  # 默认2秒
        self.verticalLayout_height_config.addWidget(self.doubleSpinBox_fall_time_window)
        
        # 高度下降阈值（米）
        self.label_fall_height_threshold = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_fall_height_threshold.setObjectName("label_fall_height_threshold")
        self.label_fall_height_threshold.setText("高度下降阈值 (m):")
        self.verticalLayout_height_config.addWidget(self.label_fall_height_threshold)
        self.doubleSpinBox_fall_height_threshold = QtWidgets.QDoubleSpinBox(self.groupBox_height_config)
        self.doubleSpinBox_fall_height_threshold.setObjectName("doubleSpinBox_fall_height_threshold")
        self.doubleSpinBox_fall_height_threshold.setMinimum(0.1)
        self.doubleSpinBox_fall_height_threshold.setMaximum(2.0)
        self.doubleSpinBox_fall_height_threshold.setSingleStep(0.1)
        self.doubleSpinBox_fall_height_threshold.setValue(0.6)  # 默认0.6米
        self.verticalLayout_height_config.addWidget(self.doubleSpinBox_fall_height_threshold)
        
        # 高度检测窗口 - 最小高度（米）
        self.label_fall_height_min = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_fall_height_min.setObjectName("label_fall_height_min")
        self.label_fall_height_min.setText("检测高度窗口 - 最小高度 (m):")
        self.verticalLayout_height_config.addWidget(self.label_fall_height_min)
        self.doubleSpinBox_fall_height_min = QtWidgets.QDoubleSpinBox(self.groupBox_height_config)
        self.doubleSpinBox_fall_height_min.setObjectName("doubleSpinBox_fall_height_min")
        self.doubleSpinBox_fall_height_min.setMinimum(-2.0)
        self.doubleSpinBox_fall_height_min.setMaximum(2.0)
        self.doubleSpinBox_fall_height_min.setSingleStep(0.1)
        self.doubleSpinBox_fall_height_min.setValue(-2.0)  # 默认-2.0米
        self.verticalLayout_height_config.addWidget(self.doubleSpinBox_fall_height_min)
        
        # 高度检测窗口 - 最大高度（米）
        self.label_fall_height_max = QtWidgets.QLabel(self.groupBox_height_config)
        self.label_fall_height_max.setObjectName("label_fall_height_max")
        self.label_fall_height_max.setText("检测高度窗口 - 最大高度 (m):")
        self.verticalLayout_height_config.addWidget(self.label_fall_height_max)
        self.doubleSpinBox_fall_height_max = QtWidgets.QDoubleSpinBox(self.groupBox_height_config)
        self.doubleSpinBox_fall_height_max.setObjectName("doubleSpinBox_fall_height_max")
        self.doubleSpinBox_fall_height_max.setMinimum(-2.0)
        self.doubleSpinBox_fall_height_max.setMaximum(2.0)
        self.doubleSpinBox_fall_height_max.setSingleStep(0.1)
        self.doubleSpinBox_fall_height_max.setValue(0.6)  # 默认0.6米
        self.verticalLayout_height_config.addWidget(self.doubleSpinBox_fall_height_max)
        
        # 清除高度按钮
        self.pushButton_clear_height = QtWidgets.QPushButton(self.groupBox_height_config)
        self.pushButton_clear_height.setObjectName("pushButton_clear_height")
        self.pushButton_clear_height.setText("清除高度")
        self.verticalLayout_height_config.addWidget(self.pushButton_clear_height)
        
        # 添加弹性空间
        self.verticalLayout_height_config.addStretch()
        
        # 组合布局
        self.horizontalLayout_height.addLayout(self.verticalLayout_height_display, 6)
        self.horizontalLayout_height.addWidget(self.groupBox_height_config, 1)
        self.verticalLayout_height.addLayout(self.horizontalLayout_height)
        
        self.tabWidget.addTab(self.tab_5, "")
        self.horizontalLayout_4.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1079, 22))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionload = QtWidgets.QAction(MainWindow)
        self.actionload.setObjectName("actionload")
        self.menu.addSeparator()
        self.menu.addAction(self.actionload)
        self.menu.addSeparator()
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.slot_init()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_11.setTitle(_translate("MainWindow", "配置"))
        self.groupBox_2.setTitle(_translate("MainWindow", "雷达配置"))
        self.label_15.setText(_translate("MainWindow", "最大距离："))
        self.label_16.setText(_translate("MainWindow", "80m"))
        self.label_36.setText(_translate("MainWindow", "最大速度："))
        self.label_14.setText(_translate("MainWindow", "3.75cm"))
        self.label_43.setText(_translate("MainWindow", "CLIPort:"))
        self.label_13.setText(_translate("MainWindow", "距离分辨率："))
        self.label_35.setText(_translate("MainWindow", "2m/s"))
        self.comboBox_7.setToolTip(_translate("MainWindow", "配置文件放到config文件夹下"))
        self.label_45.setText(_translate("MainWindow", "其他待续..."))
        self.label_12.setText(_translate("MainWindow", "配置文件："))
        self.label_44.setText(_translate("MainWindow", "DataPort:"))
        self.label_17.setText(_translate("MainWindow", "速度分辨率："))
        self.label_37.setText(_translate("MainWindow", "20m/s"))
        self.label_46.setText(_translate("MainWindow", "DataPort:"))
        self.pushButton_11.setText(_translate("MainWindow", "send"))
        self.pushButton_12.setText(_translate("MainWindow", "Exit"))
        self.groupBox_7.setTitle(_translate("MainWindow", "采集/识别"))
        self.label_5.setText(_translate("MainWindow", "color："))
        self.groupBox_3.setTitle(_translate("MainWindow", "识别"))
        self.label_2.setText(_translate("MainWindow", "model："))
        self.pushButton_15.setText(_translate("MainWindow", "recognize"))
        self.groupBox_4.setTitle(_translate("MainWindow", "采集"))
        self.label_3.setText(_translate("MainWindow", "scene："))
        self.lineEdit_6.setText(_translate("MainWindow", "chaotic_dataset"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "Back"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Dblclick"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Down"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "Front"))
        self.comboBox_3.setItemText(4, _translate("MainWindow", "Left"))
        self.comboBox_3.setItemText(5, _translate("MainWindow", "Right"))
        self.comboBox_3.setItemText(6, _translate("MainWindow", "Up"))
        self.pushButton.setText(_translate("MainWindow", "capture"))
        self.groupBox.setTitle(_translate("MainWindow", "printlog"))
        self.groupBox_9.setTitle(_translate("MainWindow", "雷达数据实时显示"))
        self.label.setText(_translate("MainWindow", "时间-距离图"))
        self.label_7.setText(_translate("MainWindow", "多普勒-时间图"))
        self.label_8.setText(_translate("MainWindow", "距离-俯仰角度图"))
        self.label_9.setText(_translate("MainWindow", "距离-方位角度图"))
        self.label_11.setText(_translate("MainWindow", "距离-多普勒图"))
        self.label_10.setText(_translate("MainWindow", "手势输出"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "real-time system"))
        # self.label_4.setText(_translate("MainWindow", "TextLabel"))

        # ZHChen 2025-05-20
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "NLOS Localization"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "点云显示"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "目标运动轨迹"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "高度检测"))
        self.groupBox_pointcloud_config.setTitle(_translate("MainWindow", "点云配置"))
        self.pushButton_test_pointcloud.setText(_translate("MainWindow", "测试点云"))






        self.menu.setTitle(_translate("MainWindow", "菜单"))
        self.actionload.setText(_translate("MainWindow", "mini"))
        self.actionload.setToolTip(_translate("MainWindow", "mini窗口"))

    def printlog(self,textEdit,string,fontcolor='blue'):
        textEdit.moveCursor(QtGui.QTextCursor.End)
        gettime = time.strftime("%H:%M:%S", time.localtime())
        textEdit.append("<font color="+fontcolor+">"+str(gettime)+"-->"+string+"</font>")

    def slot_init(self):
        self.printlog(self.textEdit, 'Welcome!',fontcolor='blue')
        self.getcolorlist()
        self.comboBox.currentIndexChanged.connect(self.setcolor)
        self.comboBox_7.arrowClicked.connect(self.configpath)           #配置文件
        self.comboBox_2.arrowClicked.connect(self.modelpath)              #选择识别模型
        self.groupBox_3.clicked.connect(lambda:self.IsRecognizeorCapture(box_name = 'box_3'))
        self.groupBox_4.clicked.connect(lambda:self.IsRecognizeorCapture(box_name = 'box_4'))
        
        self.pushButton_15.clicked.connect(lambda:self.Iscapture(self.pushButton_15,'recognizing','recognize'))
        self.pushButton.clicked.connect(lambda:self.Iscapture(self.pushButton,'capturing','capture'))
        
    def IsRecognizeorCapture(self,box_name):
        if box_name == 'box_3' and self.groupBox_3.isChecked():
            self.groupBox_4.setChecked(False)
            self.pushButton.setChecked(False)
            self.Iscapture(self.pushButton,'capturing','capture')
        elif box_name == 'box_4'and self.groupBox_4.isChecked():
            self.groupBox_3.setChecked(False)
            self.pushButton_15.setChecked(False)
            self.Iscapture(self.pushButton_15,'recognizing','recognize')
        elif box_name == 'box_3' and self.groupBox_3.isChecked()==False:
            gl.set_value('IsRecognizeorCapture',False)
            # self.printlog(self.textEdit, 'IsRecognizeorCapture:False',fontcolor='blue')
            self.graphicsView_5.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))
        elif box_name == 'box_4' and self.groupBox_4.isChecked()==False:
            gl.set_value('IsRecognizeorCapture',False)
            # self.printlog(self.textEdit, 'IsRecognizeorCapture:False',fontcolor='blue')
            self.graphicsView_5.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))

    def Iscapture(self,btn,text1,text2):
        if btn.isChecked():
            gl.set_value('IsRecognizeorCapture',True)
            # self.printlog(self.textEdit, 'IsRecognizeorCapture:True',fontcolor='blue')
            btn.setText(text1)
        else:
            gl.set_value('IsRecognizeorCapture',False)
            self.printlog(self.textEdit, 'IsRecognizeorCapture:False',fontcolor='blue')
            self.graphicsView_5.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))
            btn.setText(text2)

    def modelpath(self):
        self.comboBox_2.clear()
        self.comboBox_2.addItem("--select--")
        list = []
        if (os.path.exists(modelpathfile)):
            files = os.listdir(modelpathfile)
            for file in files:
                list.append(modelpathfile+'/'+file)
        self.comboBox_2.addItems(list)

    def configpath(self):
        self.comboBox_7.clear()
        self.comboBox_7.addItem("--select--")
        list = []
        if (os.path.exists(radarRFconfigpathfile)):
            files = os.listdir(radarRFconfigpathfile)
            for file in files:
                list.append(radarRFconfigpathfile+'/'+file)
        self.comboBox_7.addItems(list)

    def getcolorlist(self):
        # 兼容不同版本的matplotlib
        try:
            # matplotlib >= 3.5.0
            if hasattr(matplotlib, 'colormaps'):
                values = matplotlib.colormaps.keys()
            # matplotlib < 3.5.0 but >= 3.4.0
            elif hasattr(matplotlib.cm, '_cmap_registry'):
                values = matplotlib.cm._cmap_registry.keys()
            # matplotlib < 3.4.0
            elif hasattr(matplotlib.cm, 'cmap_d'):
                values = matplotlib.cm.cmap_d.keys()
            else:
                # 降级方案：使用已知的colormap列表
                values = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter']
        except:
            # 如果都失败，使用默认列表
            values = ['viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter']
        
        self.comboBox.addItem("--select--")
        self.comboBox.addItem("customize")
        for value in values:
            self.comboBox.addItem(value)

    def setcolor(self):
        if(self.comboBox.currentText()!='--select--' and self.comboBox.currentText()!=''):
            self.printlog(self.textEdit, 'selected color:'+self.comboBox.currentText(),fontcolor='blue')




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # subWin = Qt_pet()
    sys.exit(app.exec_())