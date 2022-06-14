# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AuthMainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_AuthMainWindow(object):
    def setupUi(self, AuthMainWindow):
        AuthMainWindow.setObjectName("AuthMainWindow")
        AuthMainWindow.resize(754, 560)
        self.centralwidget = QtWidgets.QWidget(AuthMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(240, 30, 20, 491))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(270, 20, 471, 501))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.cam_label = QtWidgets.QLabel(self.groupBox)
        self.cam_label.setText("")
        self.cam_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_label.setObjectName("cam_label")
        self.gridLayout_3.addWidget(self.cam_label, 1, 0, 1, 1)
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 20, 221, 504))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_4 = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.username_label_5 = QtWidgets.QLabel(self.groupBox_4)
        self.username_label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.username_label_5.setObjectName("username_label_5")
        self.gridLayout_5.addWidget(self.username_label_5, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBox_4)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.groupBox_3 = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.thumbnail_label_2 = QtWidgets.QLabel(self.groupBox_3)
        self.thumbnail_label_2.setMinimumSize(QtCore.QSize(160, 160))
        self.thumbnail_label_2.setText("")
        self.thumbnail_label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.thumbnail_label_2.setObjectName("thumbnail_label_2")
        self.gridLayout_4.addWidget(self.thumbnail_label_2, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBox_3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.MinimumExpanding)
        self.verticalLayout_4.addItem(spacerItem1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.registerFace_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.registerFace_pushButton.setObjectName("registerFace_pushButton")
        self.gridLayout.addWidget(self.registerFace_pushButton, 2, 0, 1, 2)
        self.logout_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.logout_pushButton.setObjectName("logout_pushButton")
        self.gridLayout.addWidget(self.logout_pushButton, 0, 1, 1, 1)
        self.faceAuthenticate_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.faceAuthenticate_pushButton.setObjectName("faceAuthenticate_pushButton")
        self.gridLayout.addWidget(self.faceAuthenticate_pushButton, 1, 0, 1, 2)
        self.login_pushButton = QtWidgets.QPushButton(self.groupBox_2)
        self.login_pushButton.setObjectName("login_pushButton")
        self.gridLayout.addWidget(self.login_pushButton, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        AuthMainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(AuthMainWindow)
        self.statusbar.setObjectName("statusbar")
        AuthMainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(AuthMainWindow)
        QtCore.QMetaObject.connectSlotsByName(AuthMainWindow)

    def retranslateUi(self, AuthMainWindow):
        _translate = QtCore.QCoreApplication.translate
        AuthMainWindow.setWindowTitle(_translate("AuthMainWindow", "Face Authenticator"))
        self.groupBox.setTitle(_translate("AuthMainWindow", "Camera"))
        self.groupBox_4.setTitle(_translate("AuthMainWindow", "Username"))
        self.username_label_5.setText(_translate("AuthMainWindow", "not logged in"))
        self.groupBox_3.setTitle(_translate("AuthMainWindow", "Thumbnail"))
        self.groupBox_2.setTitle(_translate("AuthMainWindow", "Operations"))
        self.registerFace_pushButton.setText(_translate("AuthMainWindow", "Register Your Face"))
        self.logout_pushButton.setText(_translate("AuthMainWindow", "Logout"))
        self.faceAuthenticate_pushButton.setText(_translate("AuthMainWindow", "Face Authentication"))
        self.login_pushButton.setText(_translate("AuthMainWindow", "Login"))
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = QtWidgets.QMainWindow()
    ui = Ui_AuthMainWindow()
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())