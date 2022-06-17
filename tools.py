# ./tools.py
# 工具函数和类
import logging
import time

import numpy
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

import mydb
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import PIL
import tempfile
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from settings import *

# face Verification function


# end faceVerification function
class FaceVerifier:
    # 初始化网络
    mtcnn = MTCNN(image_size=FACENET_INPUT_IMAGE_SIZE, margin=FACENET_INPUT_MARGIN)
    # pretrained: Either 'vggface2' or 'casia-webface'
    facenetResnet = InceptionResnetV1(pretrained='vggface2').eval()
    # todo BUG: 当输入图片没有人脸时，mtcnn返回none，对此需要调整facenet的行为，后面的函数需要改写
    # 理论上这些方法只能被一个线程调用
    # @classmethod
    # def getEmb_saveCropped(cls, img, save_path: str = None):
    #     """
    #     保存MTCNN裁剪的人脸到指定路径,返回人脸嵌入
    #     :param img: A PIL RGB Image that contains face and background
    #     :param save_path: An optional string that contains the path to save cropped image
    #     :return: A ndarray that contains the face embedding
    #     """
    #     with torch.no_grad():
    #         # Get cropped and prewhitened image tensor
    #         if save_path is None:
    #             img_cropped = cls.mtcnn(img)
    #         else:
    #             img_cropped = cls.mtcnn(img, save_path)
    #         # Calculate embedding (unsqueeze to add batch dimension)
    #         img_embedding = cls.facenetResnet(img_cropped.unsqueeze(0))
    #         img_embedding = img_embedding.numpy()
    #     return img_embedding

    @classmethod
    def getEmb_getCropped(cls, img) -> (numpy.ndarray, PIL.Image):
        """
        返回人脸嵌入和剪裁后的图片
        :param img: A PIL RGB Image that contains face and background
        :return: A tuple: (A ndarray that contains the face embedding, A PIL RGB Image that contains cropped face imaged)
        """
        with torch.no_grad():
            # Get cropped and prewhitened image tensor
            with tempfile.TemporaryDirectory() as tmpDir:
                tmpName = "tmpImg.jpg"
                img_cropped = cls.mtcnn(img, f"./{tmpDir}/{tmpName}")
                newImg = Image.open(f"./{tmpDir}/{tmpName}")
                copiedImg = newImg.copy()
                newImg = None
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = cls.facenetResnet(img_cropped.unsqueeze(0))
            img_embedding = img_embedding.numpy()
        return img_embedding, copiedImg
    @classmethod
    def isSamePersonEmb(cls, emb1, emb2) -> bool:
        """
        根据人脸嵌入距离判断是否同一个人
        :param emb1: A ndarray of face embedding
        :param emb2: A ndarray of face embedding
        :return: A bool, True if embeddings belongs to the same person
        """
        dis = np.linalg.norm(emb1 - emb2)
        return dis <= FACE_VER_THRESHOLD

class Authenticator(QObject):
    """
    Thread for faceCropping, verification and anti-spooling
    Move it to a new thread when using
    """
    need_faceVector_signal = pyqtSignal()
    authResult_signal = pyqtSignal(bool)    # todo 目前返回通过bool
    def __init__(self, faceVector, longest_wait_s = 10, accept_required = 3, frame_time = 1):
        super(Authenticator, self).__init__()
        self.longest_wait_s = longest_wait_s
        self.accept_required = accept_required
        self.frame_time = frame_time

        # 判断是否在忙的 x不用，在忙直接堵塞
        # 以下应该在每次start时重置
        # 是否通过facenet阶段
        self.verificationPass = False
        self.accept_required_left = accept_required
        self.passed_frame_storage = list()  # 通过了的thumbnail

    # # 应由start事件连接
    # def initThread(self):
    #     pass
    # todo 用于测试线程行为 现在问题是如何让这个运行在子线程
    # Qtimer挪下去？ 是的
    # Qtimermovetothread?
    def startThreadTest(self):
        # 用于间隔尝试进行MTCNN+facenet的timer
        self.authTimer = QTimer()
        # 用于判断超时的
        self.timeLimitTimer = QTimer()
        self.timeLimitTimer.setSingleShot(True)
        self.authTimer.timeout.connect(self.threadTest)
        self.authTimer.start(1)
        logging.debug(f"thread work start{QThread.currentThreadId()}")

    def threadTest(self):
        for i in range(100000):
            pass
        logging.debug(f"ThreadWorking2{QThread.currentThreadId()}")

    # 发送信号返回验证结果
    def retAuthResult(self):
        # 如果通过第一部分，尝试antispooing

        pass

    # 一次完整的认证
    def startAuth(self):
        self.timeLimitTimer.start(int(self.longest_wait_s * 1000))
        self.authTimer.start(int(self.frame_time * 1000))
        pass
    # 目前方法下认证到的人脸帧数和最后送到的人脸帧数是一样的
    # 想要增加最后活体检测用的图片数量，可在camera类中增加缓存的图片数，并加时间戳

    # 和camera连接，收到一个croppedframe后单次裁脸和facenet
    # todo 测试低分辨率下的mtcnn用时，调整认证设置
    def verifyOnce(self, arr):
        narr = np.array(arr)

        pass
# class AuthThread(QThread):
#     """
#     Thread for faceCropping, verification and anti-spooling
#     """
#     need_faceVector_signal = pyqtSignal()
#     set_faceVector_signal = pyqtSignal(numpy.ndarray)
#     # get_frame_signal = pyqtSignal()
#     def __init__(self, parent=None):
#         QThread.__init__(self, parent=parent)
#
#     # 这个线程应该进行一次完整的识别任务
#     def run(self):
#         # 先获取faceVector
#         self.need_faceVector_signal.emit()


# class CameraThread(QThread):
#     """
#     单帧图像捕获
#     接收
#     """
#     # 信号
#     change_pixmap = pyqtSignal(QImage)
#     get_faceVector_signal = pyqtSignal()
#     set_faceVector_signal = pyqtSignal(numpy.ndarray)
#
#     def __init__(self, parent=None, camera=None, capture_only=True, frame_rate=25, verify_time=0.6):
#         QThread.__init__(self, parent=parent)
#         self.camera = camera
#         self.frame_time = 1.0 / frame_rate
#         self.capture_only = capture_only
#
#     # 重写run()定义QThread行为
#     # 任务:
#     # 创建时有capture_only:不进行识别,仅充当捕获
#     # 按照帧时进行图片捕获,发送信号(由UI接收并显示)
#     # 按照处理时长间隔将裁剪框内图片送入mctnn-resnet
#     def run(self):
#         """Runs camera capture"""
#         prev = time.time()
#         while True:
#             now = time.time()
#             # 先判断是否需要，不需要则可sleep到下次开始
#             if
#             # 图像提取框
#             rval, frame = self.camera.get_frame()
#             # 转换图片格式
#             if rval:
#                 convert_qt_format = cvt_img_to_qimage(frame)
#                 qt_img = convert_qt_format.scaled(640, 480, Qt.KeepAspectRatio)
#                 if (now - prev) >= self.frame_time:
#                     # 发射信号，执行它所连接的函数
#                     self.change_pixmap.emit(qt_img)
#                     prev = now
# # #
# 更进一步这个可写成自定义控件直接放UI里
class MyQCamera(QObject):
    """
    单帧图像捕获
    接收
    """
    # 信号

    pixmap_change_signal = pyqtSignal(QPixmap)
    latest_cropped_img_asList_signal = pyqtSignal(list)
    #
    # get_faceVector_signal = pyqtSignal()
    # set_faceVector_signal = pyqtSignal(numpy.ndarray)

    def __init__(self, cam_num=0, display_size = (640, 480),cropped_frame_size=(240, 320), hint_color=(255, 0, 0), frame_rate=25):
        super(MyQCamera, self).__init__()
        self.cap = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)
        self.cam_num = cam_num
        self.display_size = display_size

        # 原始大小
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # int
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # int
        #print(f'Webcam image size: ({width}, {height})')  # webcam_size

        frame_width = cropped_frame_size[0]
        frame_height = cropped_frame_size[1]
        self.thickness = 3
        self.start_point = ((width - frame_width) // 2, (height - frame_height) // 2)
        logging.debug(f"{width} {frame_width} {height} {frame_height} {(width - frame_width) / 2}")
        logging.debug(self.start_point)
        self.end_point = (self.start_point[0] + frame_width, self.start_point[1] + frame_height)
        logging.debug(self.end_point)
        self.hint_color = hint_color
        self.frame_rate = frame_rate
        self.frame_time = 1.0 / frame_rate

        # tmpvar
        self.latestCroppedFrame = None

        # timer
        self.captureTimer = QTimer()
        self.captureTimer.timeout.connect(self.getNewFrame_QPixmap)

    def __getFrame(self):
        """Returns the next captured image array"""
        logging.info("getnewframe")
        rval, frame = self.cap.read()
        logging.debug("1")
        if rval:
            logging.debug("2")
            frame = cv2.rectangle(frame, tuple(q - self.thickness for q in self.start_point),
                                  tuple(q + self.thickness for q in self.end_point),
                                  self.hint_color, self.thickness)
            # mirrored framed
            frame = cv2.flip(frame, 1)
            # convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # saved for later
            # cropped
            cropped_img = frame.copy()
            cropped_img = cropped_img[self.start_point[1]:self.end_point[1],
                          self.start_point[0]:self.end_point[0]]
            self.latestCroppedFrame = cropped_img
            logging.debug("3")
        return rval, frame

    def getNewFrame_QPixmap(self):
        rval, frame = self.__getFrame()
        logging.debug("21")
        if rval:
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            logging.debug("22")
            # array frame to pixmap
            logging.debug(type(frame))
            logging.debug(frame)
            logging.debug(frame.shape)
            logging.debug(frame.dtype)
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            logging.debug("23")

            pixmap = QtGui.QPixmap(image).scaled(self.display_size[0], self.display_size[1],
                                                 aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            logging.debug("pixmap converted")
            self.pixmap_change_signal.emit(pixmap)
            logging.debug("pixmap send")
    def getLatestCroppedAsList(self):
        if self.latestCroppedFrame is None:
            self.latest_cropped_img_asList_signal.emit(None)
        else:
            #image = Image.fromarray(self.latestCroppedFrame)
            narr = self.latestCroppedFrame.copy()
            self.latest_cropped_img_asList_signal.emit(narr.tolist())
        pass

    def closeCamera(self):
        """
        应在使用完后调用以释放camera
        """
        self.cap.release()

    # 开始图像采集
    def start(self):
        self.captureTimer.start(int(self.frame_time * 1000))

    # 停止图像采集
    def pause(self):
        self.captureTimer.stop()