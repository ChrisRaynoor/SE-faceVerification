from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal,Qt
# import 对应需要的ui_xxx.py的窗口类
from ui_AuthMainWindow import Ui_AuthMainWindow
from run_LoginWidget import LoginWidget
# 其他依赖
import models

# 实现分离逻辑
# note
# 由于ui基于QMainWindow,这里也应对应继承.继承QWidget报错object has no attribute 'setStatusBar
class AuthMainWindow(QMainWindow, Ui_AuthMainWindow):

    def __init__(self,parent=None):
        super(AuthMainWindow, self).__init__(parent)
        self.setupUi(self)#self?
        # 该窗口的对象
        self.user = models.User()

        # 进行信号槽连接如
        # self.cameraButton.clicked.connect(self.showDialog)
        self.login_pushButton.released.connect(self.startLoginWidget)

    # 登录
    def startLoginWidget(self):
        # 判断是否已经登录
        if self.user.loggedIn:
            # 弹出提示
            QMessageBox.information(self, "Hint", "You have logged in. Do not log in again.")
            return
        # 目前看来必须将打开的窗口实例保留
        self.loginWidget = LoginWidget()
        # print('1')
        self.loginWidget.loginSignal.connect(self.login)
        # print('2')
        self.loginWidget.show()
        # print('3')
    # 示例：接收多参数信号的槽函数
    def login(self, username, password):
        if self.user.login(username, password):
            # 登录成功
            QMessageBox.information(self, "Hint", "Login success.")
        else:
            # 登陆失败
            QMessageBox.information(self, "Hint", "The username and password are invalid or do not match.")

# debug
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = AuthMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())