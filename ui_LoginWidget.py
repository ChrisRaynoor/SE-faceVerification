# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LoginWidget.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LoginWidget(object):
    def setupUi(self, LoginWidget):
        LoginWidget.setObjectName("LoginWidget")
        LoginWidget.setWindowModality(QtCore.Qt.ApplicationModal)
        LoginWidget.resize(236, 276)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(LoginWidget.sizePolicy().hasHeightForWidth())
        LoginWidget.setSizePolicy(sizePolicy)
        self.layoutWidget = QtWidgets.QWidget(LoginWidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 197, 241))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.groupBox = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.username_lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.username_lineEdit.setObjectName("username_lineEdit")
        self.gridLayout.addWidget(self.username_lineEdit, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.layoutWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.password_lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.password_lineEdit.setObjectName("password_lineEdit")
        self.gridLayout_2.addWidget(self.password_lineEdit, 0, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cancel_pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.cancel_pushButton.setObjectName("cancel_pushButton")
        self.horizontalLayout.addWidget(self.cancel_pushButton)
        self.sure_pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.sure_pushButton.setObjectName("sure_pushButton")
        self.horizontalLayout.addWidget(self.sure_pushButton)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(LoginWidget)
        QtCore.QMetaObject.connectSlotsByName(LoginWidget)

    def retranslateUi(self, LoginWidget):
        _translate = QtCore.QCoreApplication.translate
        LoginWidget.setWindowTitle(_translate("LoginWidget", "Login"))
        self.label.setText(_translate("LoginWidget", "Login"))
        self.groupBox.setTitle(_translate("LoginWidget", "Username"))
        self.groupBox_2.setTitle(_translate("LoginWidget", "Password"))
        self.cancel_pushButton.setText(_translate("LoginWidget", "Cancel"))
        self.sure_pushButton.setText(_translate("LoginWidget", "Sure"))
