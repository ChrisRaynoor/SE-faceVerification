import logging

import numpy as np
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
        # 状态变量
        self.authenticating = False
        # 该窗口的对象
        self.user = models.User()
        self.camera = tools.MyQCamera(display_size=CAM_DISPLAY_SIZE, cropped_frame_size=CAM_CROPPED_DISPLAY_SIZE,
                                      hint_color=CAM_CROPPED_DISPLAY_LINE_BGR)
        self.camera.start()
        # self.authThread = QtCore.QThread()
        self.authenticator = tools.Authenticator()

        # 要调用camera的lastfram
        self.authenticator.authTimer.timeout.connect(self.camera.getLatestCroppedAsList)
        # camera的返回信号要连接verifionce
        self.camera.latest_cropped_img_asList_signal.connect(self.authenticator.verifyOnce)
        # 登录按钮
        self.login_pushButton.released.connect(self.startLoginWidget)
        # 登出按钮
        self.logout_pushButton.released.connect(self.logout)
        # camera更新
        self.camera.pixmap_change_signal.connect(self.updateCamLabel)
        # 开始auth的行为
        self.faceAuthenticate_pushButton.released.connect(self.startAuthOnWindow)
        # auth return
        self.authenticator.authResult_signal.connect(self.showAuthResult)
        # 注册人脸
        # self.registerFace_pushButton.released.connect(self.registerFace)
    def showAuthResult(self, isPass):
        logging.debug(f"result shown at {QThread.currentThreadId()} it is ui thread?")
        QMessageBox.information(self, "Hint", f"{'pass' if isPass else 'notpass'}")
    # 开始验证的一系列行为
    def startAuthOnWindow(self):
        # 验证登录
        if not self.user.loggedIn:
            QMessageBox.information(self, "Hint", "Please login first.")
            return
        # 确认有人脸
        if self.user.getFaceVector() is None:
            QMessageBox.information(self, "Hint", "Please register face first")
            return
        logging.debug(f"ui thread is {QThread.currentThreadId()}")
        # self.camera.start()
        faceVector = self.user.getFaceVector()
        # logging.debug(type(faceVector))
        # logging.debug(faceVector)
        self.authenticator.startAuth(self.user.getFaceVector())
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
            self.username_label_5.setText(self.user.username)
            QMessageBox.information(self, "Hint", "Login success.")
        else:
            # 登陆失败
            QMessageBox.information(self, "Hint", "The username and password are invalid or do not match.")

    # 登出按钮槽
    def logout(self):
        # 判断是否在认证
        if self.authenticating:
            QMessageBox.information(self, "Hint", "Face authentication in progress, please wait.")
            return
        # 判断是否已经登录
        if not self.user.loggedIn:
            # 未登录
            QMessageBox.information(self, "Hint", "Please login first.")
            return
        # 已登录
        reply = QMessageBox.question(self, "Logout", "Are you sure to log out?")
        # 是否确定登出
        if reply == QMessageBox.Yes:
            self.user.logout()
            self.username_label_5.setText("not logged in")
        # else: do nothing
    # # 注册人脸槽
    # def registerFace(self):
    #     "this is for demonstration purpose"
    #     # 判断是否已经登录
    #     if not self.user.loggedIn:
    #         # 未登录
    #         QMessageBox.information(self, "Hint", "Please login first.")
    #         return
    #     # 是否在验证
    #     if self.authenticating:
    #         QMessageBox.information(self, "Hint", "Face authentication in progress, please wait")
    #         return
    #     # 判断是否已注册
    #     if self.user.getFaceVector() is not None:
    #         # 已注册
    #         QMessageBox.information(self, "Hint", "The account's face information has been registered.\n"
    #                                               "Please contact the administrator to update face information.")
    #         return
    #     # 未注册
    #     frame = self.camera.getLatestCroppedAsList()
    #     image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
    #     # logging.debug("23")
    #     pixmap = QtGui.QPixmap(image).scaled(self.display_size[0], self.display_size[1],
    #                                          aspectRatioMode=QtCore.Qt.KeepAspectRatio)
    #     # 显示thumbnail
    #     self.thumbnail_label_2.setPixmap(pixmap)
    #     reply = QMessageBox.question(self, "Register", "Thumbnail is showed on the left.\n"
    #                                                  "Are you sure to register with the displayed face?")
    #     if reply == QMessageBox.No:#不同意,donothing
    #         return
    #     # 同意
    #     frameForEmb = np.array(frame).astype(np.uint8)
    #     emb, _ = tools.FaceVerifier.get_emb_and_cropped_from_np(frameForEmb)
    #     # 是否有人脸
    #     if emb is None:
    #         # 无人脸
    #         QMessageBox.information(self, "Hint", "No face detected. Please try again.")
    #         return
    #     # 有人脸
    #     # 推荐先验证
    #     # reply = QMessageBox.question(self, "Register", "Do you want to confirm the validity of the face?"
    #     #                                                "Press Yes to start test run, or press No to skip.")
    #     # if reply == QMessageBox.Yes:
    #     self.user.setFaceVector(emb)
    #     QMessageBox.information(self, "Hint", "Register success.")

# debug
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = AuthMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())