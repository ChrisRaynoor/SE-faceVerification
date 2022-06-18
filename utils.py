import logging

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def stream_process(path, step: int, gray: bool, net, out_path: str, saved=0):
    """
    将视频处理成帧
    :param out_path: 输出路径
    :param saved: 已保存图片数
    :param gray: 是否转化为灰度图，False时转化为HSV或YCbCr
    :param step: 保存帧数的间隔
    :param path: 视频路径
    """

    print("framing video {} ...".format(path))
    video = cv2.VideoCapture(path)
    # 已读入帧数
    read = 0
    while True:
        (success, frame) = video.read()
        if not success:
            break

        read += 1
        if read % step != 0:
            continue
        face, coordinate = face_detect(frame, (300, 300), net, 0.5)
        if face is not None:
            out = out_path + "/{}.png".format(saved)
            saved += 1
            print("saving {}".format(out))
            cv2.imwrite(out, face)
        cv2.imshow("video", frame)
        print('....')
        if cv2.waitKey(5) is 27:  # 按键盘的ESC键可退出，同时停顿一微秒
            break
        if cv2.getWindowProperty('video', cv2.WND_PROP_AUTOSIZE) < 1:  # 用鼠标点击窗口退出键实现退出循环
            break
    video.release()
    return saved


def face_detect(frame, size, net, prob):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, dsize=size), 1.0, size, (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > prob:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            return face, (startX, startY, endX, endY)
    return None, None


def face2gray(frame, size):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, size)
    return frame


def face2ycbcr(frame, size):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    frame = cv2.resize(frame, size)
    return frame


def generateFeature(in_path, num_data, isReal=False, isGray=True):
    """
    生成训练数据
    :param isGray:
    :param in_path:脸部数据路径
    :param num_data:数据量
    """
    sample = []
    for i in range(num_data):
        path = in_path + "/{}.png".format(i)
        print("processing fig {}".format(path))
        img = cv2.imread(path)
        if isGray:
            face = face2gray(img, (64, 64))
            lbp = upLBP(face)
            sample.append(LBPH(lbp, 59))
        else:
            face = face2ycbcr(img, (64, 64))
            y_lbp = LBPH(upLBP(face[:, :, 0]), 59)
            cb_lbp = LBPH(upLBP(face[:, :, 1]), 59)
            cr_lbp = LBPH(upLBP(face[:, :, 2]), 59)
            lbp = np.concatenate((y_lbp, cb_lbp, cr_lbp))
            sample.append(lbp)
    sample = np.array(sample)
    if isReal:
        labels = np.ones(sample.shape[0])
    else:
        labels = np.zeros(sample.shape[0])
    return sample, labels


def LBPH(lbp, num_patterns):
    data = np.reshape(lbp, (-1))
    bins = num_patterns - 1
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    return hist


def upLBP(img):
    radius = 3
    neighbor = 8

    h, w = img.shape
    values = np.zeros((h - 2 * radius, w - 2 * radius), dtype=img.dtype)
    table = createTable(img)
    for k in range(neighbor):
        # 偏移量，不一定为整数
        dx = radius * np.cos(2.0 * np.pi * k / neighbor)
        dy = radius * np.sin(2.0 * np.pi * k / neighbor)

        fx = int(np.floor(dx))
        cx = int(np.ceil(dx))
        fy = int(np.floor(dy))
        cy = int(np.ceil(dy))

        tx = dx - fx
        ty = dy - fy
        # 双插值权重
        w1 = (1 - tx) * (1 - ty)  # fx fy
        w2 = (1 - tx) * ty  # fx cy
        w3 = tx * (1 - ty)  # cx fy
        w4 = tx * ty  # cx cy

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = img[i, j]
                value = img[i + fy, j + fx] * w1 + img[i + cy, j + fx] * w2 + img[i + fy, j + cx] * w3 + img[
                    i + cy, j + cx] * w4
                values[i - radius, j - radius] |= (value > center) << (neighbor - k - 1)
                if (k + 1) % neighbor == 0:
                    values[i - radius, j - radius] = table[values[i - radius, j - radius]]
    return values


def getShift(number: int):
    """
    计算跳变次数
    :param number: 输入数字
    """
    count = 0
    # 将number化为8位二进制数
    binary = "{0:0>8b}".format(number)
    for i in range(1, len(binary)):
        if binary[i] != binary[i - 1]:
            count += 1
    return count


def createTable(img):
    """
    创建uniform pattern LBP的表
    """
    table = np.zeros(256, img.dtype)
    value = 0
    for i in range(256):
        if getShift(i) < 3:
            value += 1
            table[i] = value
    return table


def generateDataSet(idx_in_path, camera_type=None, attack_type=None, isTest=False, net=None):
    """
    获取训练集或测试集
    :param idx_in_path:
    :param camera_type:
    :param attack_type:
    :param isTest:
    """
    file = open(idx_in_path)
    index = []
    for line in file.readlines():
        index.append(line.strip('\n'))
    saved_real = 0
    postfix = ['mp4', 'mov']
    if isTest:
        out = "dataset/test"
    else:
        out = "dataset/train"
    out_real_path = out + "/real"
    out_attack_path = out + "/attack"
    for idx in index:
        for camera in camera_type:
            if camera == 'android':
                post = postfix[0]
            else:
                post = postfix[1]
            in_path = "dataset/scene01/real/real_client0{}_{}_SD_scene01.{}".format(idx, camera, post)
            saved_real = stream_process(in_path, 15,
                                        False,
                                        net, out_real_path, saved_real)
    saved_attack = 0
    for idx in index:
        for camera in camera_type:
            if camera == 'android':
                post = postfix[0]
            else:
                post = postfix[1]
            for attack in attack_type:
                in_path = "dataset/scene01/attack/attack_client0{}_{}_SD_{}_scene01.{}".format(idx, camera, attack,
                                                                                               post)
                saved_attack = stream_process(in_path, 15,
                                              False,
                                              net, out_attack_path, saved_attack)


def metric(predict_proba, labels):
    predict = np.greater(predict_proba[:, 1], 0.5)
    tn, fp, fn, tp = confusion_matrix(labels, predict).flatten()
    acc = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn)  # apcer
    frr = fn / (tp + fn)  # bpcer
    hter = (far + frr) / 2  # acer
    print("false accept rate: {}".format(far))
    print("false reject rate: {}".format(frr))

    fpr, tpr, threshold = roc_curve(labels, predict_proba[:, 1])
    auc_v = auc(fpr, tpr)  # area under curve
    dist = abs((1 - fpr) - tpr)
    eer = fpr[np.argmin(dist)]
    plt.plot(fpr, tpr, label='area under curve(auc): %0.2f' % auc_v)
    plt.plot([0, 1], [1, 0])
    plt.plot([eer, eer], [0, tpr[np.argmin(dist)]], label='@EER', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.legend()
    plt.savefig("roc.png")
    return acc, eer, hter


def judge(result):
    acc = float(np.sum(result == 1)) / result.size
    if acc > 0.7:
        return True, acc
    else:
        return False, 1 - acc


def anti_spoofing(imgs, svm):
    samples = []
    logging.debug("5000")
    for img in imgs:
        logging.debug("5001")
        face = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        logging.debug("5002")
        face = cv2.resize(face, (64, 64))
        logging.debug("5003")
        y_lbp = LBPH(upLBP(face[:, :, 0]), 59)
        cb_lbp = LBPH(upLBP(face[:, :, 1]), 59)
        cr_lbp = LBPH(upLBP(face[:, :, 2]), 59)
        logging.debug("5004")
        lbp = np.concatenate((y_lbp, cb_lbp, cr_lbp))
        logging.debug("5005")
        samples.append(lbp)
        logging.debug("5006")
    samples = np.array(samples)
    logging.debug("5007")
    predict_proba = svm.predict_proba(samples)
    logging.debug("5008")
    predict = svm.predict(samples)
    logging.debug("5009")
    print(predict_proba)
    print(predict)
    return judge(predict)
