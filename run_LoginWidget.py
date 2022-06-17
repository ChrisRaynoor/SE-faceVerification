from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal,Qt
# import 对应需要的ui_xxx.py的窗口类
from ui_LoginWidget import Ui_LoginWidget
from logging import debug
# 其他依赖
import models

# 实现分离逻辑
class LoginWidget(QWidget, Ui_LoginWidget):
    # 定义信号,用于传递参数给主窗口
    loginSignal = pyqtSignal([str, str])

    # 类变量
    saved_username = ""
    def __init__(self,parent=None):
        super(LoginWidget, self).__init__(parent)
        self.setupUi(self)#self?
        # 该窗口的对象
        self.username_lineEdit.setText(LoginWidget.saved_username)
        # 进行信号槽连接
        # 连接退出按钮
        self.cancel_pushButton.released.connect(self.close)
        # 连接确定
        self.sure_pushButton.released.connect(self.emitLoginSignal)
        # print("i")
    @classmethod
    # 修改saved_username
    def setSaved_username(cls, username):
        cls.saved_username = username

    # 发出longin信号
    def emitLoginSignal(self):
        # print('emit')
        username = self.username_lineEdit.text()
        password = self.password_lineEdit.text()
        self.hide()
        self.loginSignal[str, str].emit(username, password)
        self.close()
        # print('closed')
        # debug:说明close在这里不代表其被删除
        # self.show()
    # todo 如果要实现保存上次的登录名，使用closeevent