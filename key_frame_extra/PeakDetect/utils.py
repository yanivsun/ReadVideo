import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def scale(img, xScale, yScale):
    '''
    使用区域插值法进行图像缩放，这种方法在缩小图像时效果较好，可以减少锯齿和模糊。
    :param img:
    :param xScale: 0-1
    :param yScale: 0-1
    :return:
    '''
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res

def crop(infile, height, width):
    '''
    裁剪图像：将图像裁剪成多个小块
    :param infile: 图像路径
    :param height:
    :param width:
    :return:
    '''
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)

def averagePixels(path):
    '''
    计算给定图像的平均RGB值
    :param path:
    :return: RGB通道图像平均值，像素总数
    '''
    r, g, b = 0, 0, 0
    count = 0
    pic = Image.open(path)
    for x in range(pic.size[0]):
        for y in range(pic.size[1]):
            imgData = pic.load()
            tempr, tempg, tempb = imgData[x, y]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    return (r / count), (g / count), (b / count), count


def convert_frame_to_grayscale(frame):
    '''
    将输入的图像帧转换为灰度图像，并对其进行模糊处理
    :param frame:
    :return: 模糊前后的图像
    '''
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray

def prepare_dirs(keyframePath, imageGridsPath, csvPath):
    '''
    创建指定的文件夹
    :param keyframePath:用于存储关键帧的目录路径
    :param imageGridsPath:用于存储图像网格的目录路径
    :param csvPath:用于存储CSV文件的目录路径
    :return:
    '''
    if not os.path.exists(keyframePath):
        os.makedirs(keyframePath)
    if not os.path.exists(imageGridsPath):
        os.makedirs(imageGridsPath)
    if not os.path.exists(csvPath):
        os.makedirs(csvPath)

def plot_metrics(indices, lstfrm, lstdiffMag):
    '''
    绘制像素差异的图形
    :param indices: 特定帧的索引
    :param lstfrm: 帧数
    :param lstdiffMag: 每帧之间的像素差异值
    :return:
    '''
    y = np.array(lstdiffMag)
    plt.plot(indices, y[indices], "x")
    l = plt.plot(lstfrm, lstdiffMag, 'r-')
    plt.xlabel('frames')
    plt.ylabel('pixel difference')
    plt.title("Pixel value differences from frame to frame and the peak values")
    plt.show()