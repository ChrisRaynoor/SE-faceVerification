# ./tools.py
# 工具函数和类
import logging
import time

import numpy
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer, QMutex
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

# 用于保护网络资源的mutex todo（暂不确定mtcnn和facenet的线程安全性）
mutex = QMutex()

class FaceVerifier:
    # 初始化网络
    mtcnn = MTCNN(image_size=FACENET_INPUT_IMAGE_SIZE, margin=FACENET_INPUT_MARGIN)
    # pretrained: Either 'vggface2' or 'casia-webface'
    facenetResnet = InceptionResnetV1(pretrained='vggface2').eval()
    # todo BUG: 当输入图片没有人脸时，mtcnn返回none，对此需要调整facenet的行为，后面的函数需要改写
    # 理论上这些方法只能被一个线程调用
    @classmethod
    def get_emb_and_cropped_from_np(cls, frame):
        """
        裁剪和识别
        返回（特征，裁剪图），均为ndarray
        没有人脸则均为none
        """
        with torch.no_grad():
            # Get cropped and prewhitened image tensor
            with tempfile.TemporaryDirectory() as tmpDir:
                tmpName = "tmpImg.jpg"
                img_cropped = cls.mtcnn(frame, f"./{tmpDir}/{tmpName}")
                # 验证是否有人脸
                if img_cropped is None: # 无人脸
                    return None, None
                # 有人脸
                else:
                    newImg = Image.open(f"./{tmpDir}/{tmpName}")
                    copiedImg = newImg.copy()
                    # 转为ndarray
                    resultArr = np.asarray(copiedImg)
                    newImg = None
            # Calculate embedding (unsqueeze to add batch dimension)
            img_embedding = cls.facenetResnet(img_cropped.unsqueeze(0))
            img_embedding = img_embedding.numpy()
        return img_embedding, resultArr
        pass
    # @classmethod
    # def getEmb_getCropped(cls, img) -> (numpy.ndarray, PIL.Image):
    #     """
    #     返回人脸嵌入和剪裁后的图片
    #     :param img: A PIL RGB Image that contains face and background
    #     :return: A tuple: (A ndarray that contains the face embedding, A PIL RGB Image that contains cropped face imaged)
    #     """
    #     with torch.no_grad():
    #         # Get cropped and prewhitened image tensor
    #         with tempfile.TemporaryDirectory() as tmpDir:
    #             tmpName = "tmpImg.jpg"
    #             img_cropped = cls.mtcnn(img, f"./{tmpDir}/{tmpName}")
    #             newImg = Image.open(f"./{tmpDir}/{tmpName}")
    #             copiedImg = newImg.copy()
    #             newImg = None
    #         # Calculate embedding (unsqueeze to add batch dimension)
    #         img_embedding = cls.facenetResnet(img_cropped.unsqueeze(0))
    #         img_embedding = img_embedding.numpy()
    #     return img_embedding, copiedImg

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
    def __init__(self, camera, longest_wait_s = 10, accept_required = 3, frame_time = 1):
        super(Authenticator, self).__init__()
        self.longest_wait_s = longest_wait_s
        self.accept_required = accept_required
        self.frame_time = frame_time
        # camera
        self.camera = camera    #MyQCamera

        # QTimer
        # 用于间隔尝试进行MTCNN+facenet的timer
        self.authTimer = QTimer()

        # todo 现测试camera作为参数传入是否可行，作为对象应该引用？不行试试套个list.此处强耦合要求camera
        # 要调用camera的lastfram
        self.authTimer.timeout.connect(self.camera.getLatestCroppedAsList)
        # camera的返回信号要连接verifionce
        self.camera

        # 用于判断超时的timer
        self.timeLimitTimer = QTimer()
        self.timeLimitTimer.setSingleShot(True)
        self.timeLimitTimer.timeout.connect(self.retAuthResult)
        # 以下应该在每次完整任务前start时重置
        # 是否通过facenet阶段
        self.faceVector = None
        self.verificationPass = False
        self.accept_required_left = self.accept_required
        self.passed_frame_storage = list()  # 通过了的thumbnail

    # 发送信号返回验证结果
    # 两种情况:
    # 1. facenet部分超时,直接返回不通过
    # 2. facenet部分通过,进行antispooling,返回是否通过
    def retAuthResult(self):
        # 如果通过第一部分，尝试antispooing

        pass

    # todo testing
    # 一次完整的认证,同样应该用槽或者invoke method调用?
    # now testing: 槽连接
    def startAuth(self, faceVector):
        # 重置控制量
        self.faceVector = faceVector
        self.verificationPass = False
        self.accept_required_left = self.accept_required
        self.passed_frame_storage = list()  # 通过了的thumbnail
        # logging.debug(f"auth work at {QThread.currentThreadId()}")
        # 开始计时
        self.timeLimitTimer.start(int(self.longest_wait_s * 1000))
        self.authTimer.start(int(self.frame_time * 1000))
        pass
    # 目前方法下认证到的人脸帧数和最后送到的人脸帧数是一样的
    # 想要增加最后活体检测用的图片数量，可在camera类中增加缓存的图片数，并加时间戳

    # 和camera连接，收到一个croppedframe后单次裁脸和facenet
    # 创建新线程进行一张图片任务
    # todo 测试低分辨率下的mtcnn用时，调整认证设置
    def verifyOnce(self, inputFrame):
        logging.debug(f"authOnce start at {QThread.currentThreadId()}")
        frame = np.array(inputFrame)
        # 创建新的工作类并移到线程开始工作
        worker = VerificationWorker(self.faceVector, frame)

        pass

# 人脸裁剪加验证worker
class VerificationWorker(QObject):
    """
    人脸裁剪加验证worker
    输入:通过构造函数获取facevector(?needtest)
    输出:通过信号返回人脸是否通过,通过人脸的剪裁array
    """
    # 信号
    retVerification_signal = pyqtSignal(bool, list)

    def __init__(self, faceVector, frame):
        super(VerificationWorker, self).__init__()
        self.faceVector = faceVector
        self.frame = frame
    def runTask(self):
        """
        执行长任务,由于连接start,理论上没有实参?
        """
        # 保护网络
        mutex.lock()
        emb, croppedFrame = FaceVerifier.get_emb_and_cropped_from_np(frame=self.frame)
        logging.debug(f"verify task working at {QThread.currentThreadId()}, faceVector{self.faceVector[0,0:2]}, frame{self.frame[0, 0:3]}")
        # 判断是否是人脸
        if croppedFrame is None:    # 非人脸直接返回
            self.retVerification_signal.emit(False, None)
        else:
            # 是人脸先验证
            res = FaceVerifier.isSamePersonEmb(emb, self.faceVector)
            # 判断是否通过
            if not res:     # 不通过
                self.retVerification_signal.emit(False, None)
            else:
                # 转换array才能发送信号
                arr = croppedFrame.tolist()
                self.retVerification_signal.emit(True, arr)
        # 要解除保护，所以不能直接return，除非tryfinally
        mutex.unlock()
        logging.debug(f"verify task return")
    pass
# antispooling worker
class AntiSpoolingWorker(QObject):
    pass

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
        # logging.debug(f"{width} {frame_width} {height} {frame_height} {(width - frame_width) / 2}")
        # logging.debug(self.start_point)
        self.end_point = (self.start_point[0] + frame_width, self.start_point[1] + frame_height)
        # logging.debug(self.end_point)
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
        # logging.info("getnewframe")
        rval, frame = self.cap.read()
        # logging.debug("1")
        if rval:
            # logging.debug("2")
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
            # logging.debug("3")
        return rval, frame

    def getNewFrame_QPixmap(self):
        logging.debug(f"camera working at {QThread.currentThreadId()}")
        rval, frame = self.__getFrame()
        # logging.debug("21")
        if rval:
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
            # logging.debug("22")
            # array frame to pixmap
            # logging.debug(type(frame))
            # logging.debug(frame)
            # logging.debug(frame.shape)
            # logging.debug(frame.dtype)
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            # logging.debug("23")

            pixmap = QtGui.QPixmap(image).scaled(self.display_size[0], self.display_size[1],
                                                 aspectRatioMode=QtCore.Qt.KeepAspectRatio)
            # logging.debug("pixmap converted")
            self.pixmap_change_signal.emit(pixmap)
            # logging.debug("pixmap send")
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
        logging.debug(f"camera started at {QThread.currentThreadId()}")

    # 停止图像采集
    def pause(self):
        self.captureTimer.stop()