# ./tools.py
# 工具函数
import numpy

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

# 初始化网络
mtcnn = MTCNN(image_size=FACENET_INPUT_IMAGE_SIZE, margin=FACENET_INPUT_MARGIN)
# pretrained: Either 'vggface2' or 'casia-webface'
facenetResnet = InceptionResnetV1(pretrained='vggface2').eval()

def getEmb_saveCropped(img, save_path: str = None):
    """
    保存MTCNN裁剪的人脸到指定路径,返回人脸嵌入
    :param img: A PIL RGB Image that contains face and background
    :param save_path: An optional string that contains the path to save cropped image
    :return: A ndarray that contains the face embedding
    """
    with torch.no_grad():
        # Get cropped and prewhitened image tensor
        if save_path is None:
            img_cropped = mtcnn(img)
        else:
            img_cropped = mtcnn(img, save_path)
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = facenetResnet(img_cropped.unsqueeze(0))
        img_embedding = img_embedding.numpy()
    return img_embedding
def getEmb_getCropped(img) -> (numpy.ndarray, PIL.Image):
    """
    返回人脸嵌入和剪裁后的图片
    :param img: A PIL RGB Image that contains face and background
    :return: A tuple: (A ndarray that contains the face embedding, A PIL RGB Image that contains cropped face imaged)
    """
    with torch.no_grad():
        # Get cropped and prewhitened image tensor
        with tempfile.TemporaryDirectory() as tmpDir:
            tmpName = "tmpImg.jpg"
            img_cropped = mtcnn(img, f"./{tmpDir}/{tmpName}")
            newImg = Image.open(f"./{tmpDir}/{tmpName}")
            copiedImg = newImg.copy()
            newImg = None
        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = facenetResnet(img_cropped.unsqueeze(0))
        img_embedding = img_embedding.numpy()
    return img_embedding, copiedImg
def isSamePersonEmb(emb1, emb2) -> bool:
    """
    根据人脸嵌入距离判断是否同一个人
    :param emb1: A ndarray of face embedding
    :param emb2: A ndarray of face embedding
    :return: A bool, True if embeddings belongs to the same person
    """
    dis = np.linalg.norm(emb1 - emb2)
    return dis <= FACE_VER_THRESHOLD