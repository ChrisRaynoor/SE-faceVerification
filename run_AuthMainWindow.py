import logging

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt, QThread
# import 对应需要的ui_xxx.py的窗口类
import tools
from ui_AuthMainWindow import Ui_AuthMainWindow
from run_LoginWidget import LoginWidget
# 其他依赖
import models
from settings import *
# 实现分离逻辑
# note
# 由于ui基于QMainWindow,这里也应对应继承.继承QWidget报错object has no attribute 'setStatusBar
class AuthMainWindow(QMainWindow, Ui_AuthMainWindow):

    def __init__(self,parent=None):
        super(AuthMainWindow, self).__init__(parent)
        self.setupUi(self)#self?
        # logging.debug(f"main thread {QThread.currentThreadId()}")
        # 该窗口的对象
        self.user = models.User()
        self.camera = tools.MyQCamera(display_size=CAM_DISPLAY_SIZE, cropped_frame_size=CAM_CROPPED_DISPLAY_SIZE,
                                      hint_color=CAM_CROPPED_DISPLAY_LINE_BGR)

        # self.authThread = QtCore.QThread()
        self.authenticator = tools.Authenticator()
        # self.authenticator.moveToThread(self.authThread)
        # 直接开始工作 作为后台守护线程
        # self.authThread.started.connect(self.authenticator.initThread)
        # self.authThread.start()

        # 进行信号槽连接如
        # self.cameraButton.clicked.connect(self.showDialog)
        # 登录按钮
        self.login_pushButton.released.connect(self.startLoginWidget)
        # 登出按钮
        self.logout_pushButton.released.connect(self.logout)
        # camera更新
        self.camera.pixmap_change_signal.connect(self.updateCamLabel)
        # todo: 现用于测试的原按钮 测试2 直接使用信号调用
        # 开始auth的行为
        self.faceAuthenticate_pushButton.released.connect(self.camera.start)
        self.faceAuthenticate_pushButton.released.connect(self.authenticator.startAuth)

    # 摄像更新
    def updateCamLabel(self, pixmap):
        # logging.debug("try set pixmap")
        self.cam_label.setPixmap(pixmap)
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

    # 登出按钮槽
    def logout(self):
        # 判断是否已经登录
        if self.user.loggedIn:
            # 已登录
            reply = QMessageBox.question(self, "Logout", "Are you sure to log out?")
            # 是否确定登出
            if reply == QMessageBox.Yes:
                self.user.logout()
            # else: do nothing
            pass
        else:
            # 未登录
            QMessageBox.information(self, "Hint", "Please login first.")
# debug
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = AuthMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())