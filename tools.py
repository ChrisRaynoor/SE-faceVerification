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

    @classmethod
    def getEmb_saveCropped(cls, img, save_path: str = None):
        """
        保存MTCNN裁剪的人脸到指定路径,返回人脸嵌入
        :param img: A PIL RGB Image that contains face and background
        :param save_path: An optional string that contains the path to save cropped image
        :return: A ndarray that contains the face embedding
        """
        with torch.no_grad():
            # Get cropped and prewhitened image tensor
            if save_path is None:
                img_cropped = cls.mtcnn(img)
            else:
                img_cropped = cls.mtcnn(img, save_path)
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = cls.facenetResnet(img_cropped.unsqueeze(0))
            img_embedding = img_embedding.numpy()
        return img_embedding

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

# class Camera:
#     """Captures image from selected camera and returns image inside drawed rectangle"""
#     def __init__(self, cam_num=0, cropped_frame_size=(240, 320)):
#         """
#
#         :param cam_num: camera index, 0 for default camera
#         :param cropped_frame_size: cropped frame size for camera frame
#         """
#         self.cap = cv2.VideoCapture(cam_num)
#         self.cam_num = cam_num
#         width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # int
#         height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # int
#         print(f'Webcam image size: ({width}, {height})')  # webcam_size
#         frame_width = cropped_frame_size[0]
#         frame_height = cropped_frame_size[1]
#         self.thickness = 3
#         self.start_point = ((width - frame_width) // 2, (height - frame_height) // 2)
#         self.end_point = (self.start_point[0] + frame_width, self.start_point[1] + frame_height)
#         self.color = (255, 0, 0)
#
#     def get_frame(self):
#         """Returns the next captured image array"""
#         rval, frame = self.cap.read()
#         if rval:
#             frame = cv2.rectangle(frame, tuple(q-self.thickness for q in self.start_point),
#                                   tuple(q+self.thickness for q in self.end_point),
#                                   self.color, self.thickness)
#             # mirrored framed
#             frame = cv2.flip(frame, 1)
#             # convert to RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         return rval, frame
#
# # # todo: validate cropImage()
# #     def cropImage(self, img):
# #         """
# #         get image array slice according to current Camera settings
# #         :param img: original image
# #         :return: original image's slice
# #         """
# #         cropped_img = img[self.start_point[1]:self.end_point[1],
# #                       self.start_point[0]:self.end_point[0]]
# #         return cropped_img
#
#     def close_camera(self):
#         """Releases camera"""
#         self.cap.release()


# class AuthThread(QThread):
#     """
#     Thread for faceCropping, verification and anti-spooling
#     """
#     get_faceVector_signal = pyqtSignal()
#     set_faceVector_signal = pyqtSignal(numpy.ndarray)
#     # get_frame_signal = pyqtSignal()
#     def __init__(self, parent=None):
#         QThread.__init__(self, parent=parent)

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
    # latest_cropped_PIL_signal = pyqtSignal(PIL.Image) # todo 类型不支持
    #
    # get_faceVector_signal = pyqtSignal()
    # set_faceVector_signal = pyqtSignal(numpy.ndarray)

    def __init__(self, cam_num=0, display_size = (640, 480),cropped_frame_size=(240, 320), hint_color=(255, 0, 0), frame_rate=25):
        super(MyQCamera, self).__init__()
        self.cap = cv2.VideoCapture(cam_num)
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
    # todo 类型不支持
    # def getLatestCroppedPIL(self):
    #     if self.latestCroppedFrame is None:
    #         self.latest_cropped_PIL_signal.emit(None)
    #     else:
    #         image = Image.fromarray(self.latestCroppedFrame)
    #         self.latest_cropped_PIL_signal.emit(image)
    #     pass

    def closeCamera(self):
        """
        应在使用完后调用以释放camera
        """
        self.cap.release()

    # 开始图像采集
    def start(self):
        self.captureTimer.start(self.frame_time * 1000)

    # 停止图像采集
    def pause(self):
        self.captureTimer.stop()