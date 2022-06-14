# ./tools.py
# 工具函数
import mydb
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from settings import *


# 初始化网络
mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=MARGIN)
# pretrained: Either 'vggface2' or 'casia-webface'
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def getEmb(img):
    '''
    获取人脸嵌入
    :param img:RGB PIL Img 未截取人脸的图片
    :return: 256-D np.array 人脸嵌入
    '''
    with torch.no_grad():
        # Get cropped and prewhitened image tensor
        img_cropped = mtcnn(img, "tmpImg.jpg")
        newImg = Image.open("tmpImg.jpg")
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = resnet(img_cropped.unsqueeze(0))
        img_embedding = img_embedding.numpy()
    return img_embedding

def isSamePersonEmb(emb1, emb2):
    dis = np.linalg.norm(emb1 - emb2)
    return dis <= FACE_VER_THRESHOLD