import os
import cv2
import csv
import numpy as np
import time
import peakutils
from utils import convert_frame_to_grayscale, prepare_dirs, plot_metrics

def keyframeDetection(source, dest, Thres, plotMetrics=False, verbose=False):
    keyframePath = dest + '/keyFrames'
    imageGridsPath = dest + '/imageGrids'
    csvPath = dest + '/csvFile'
    path2file = csvPath + '/output.csv'
    prepare_dirs(keyframePath, imageGridsPath, csvPath)

    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = [] #存储每帧的帧号
    lstdiffMag = [] #存储相邻帧之间的像素差异值
    timeSpans = [] #存储处理每帧所需的时间
    images = [] #存储灰度图像
    full_color = [] #存储原始彩色图像
    lastFrame = None #存储前一帧的模糊灰度图像
    Start_time = time.process_time()

    for i in range(length):
        ret, frame = cap.read()
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        # 当前帧号
        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time - Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

    cap.release()

    #计算像素差异的基线和峰值
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y - base, Thres, min_dist=1)

    #plot
    if (plotMetrics):
        plot_metrics(indices, lstfrm, lstdiffMag)

    cnt = 1
    for x in indices:
        cv2.imwrite(os.path.join(keyframePath, 'keyframe' + str(cnt) + '.jpg'), full_color[x])
        cnt += 1
        log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
        if (verbose):
            print(log_message)
        with open(path2file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(log_message)
            csvFile.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # source = "./acrobacia.mp4"
    source = r"E:\Retrievl\ReadVideo\tmp_data\videos\Base jumping.mp4"
    dest = "./tmpdata"
    Thres = 0.5
    keyframeDetection(source,dest,Thres,True,True)