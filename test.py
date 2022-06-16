# ./main.py
# 程序入口:开启Qt程序和创建主窗口
import cv2
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
    logging.basicConfig(filename=f'test{timestr}.log', level=logging.DEBUG)
    logging.info('Started')
    # qt程序
    cap = cv2.VideoCapture(0)
    rval, frame = cap.read()
    # convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    logging.debug(frame.shape)
    logging.debug(frame.dtype)
    logging.debug(frame)
    image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
    cap.release()
    # end qt程序
    logging.info('Finished')
if __name__ == "__main__":
    main()