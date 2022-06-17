# ./main.py
# 程序入口:开启Qt程序和创建主窗口
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import run_AuthMainWindow
# 用于调试信息
import logging
import time


def main():
    # 配置logging输出
    # 最终版应该改为 level=logging.WARNING
    timestamp = time.time()
    timestr = time.strftime("-%Y-%m-%d--%H-%M-%S-", time.localtime()) + str(int(timestamp))
    logging.basicConfig(filename=f'main{timestr}.log', level=logging.DEBUG)
    logging.info('Started')
    # qt程序
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = run_AuthMainWindow.AuthMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
    # end qt程序
    logging.info('Finished')
if __name__ == "__main__":
    main()