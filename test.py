import os
from pdf2image import convert_from_path
import cv2
import fitz  # PyMuPDF
import numpy as np
#ref:
# https://github.com/0ssamaak0/CLIPPyX

def pdf_to_images(pdf_path, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 将 PDF 转换为图像
    images = convert_from_path(pdf_path)

    # 保存每一页为图像文件
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i + 1}.png')
        image.save(image_path, 'PNG')
        print(f'Saved: {image_path}')

class VideoStreamer(object):
    """ Class to help process image streams. Three types of possible inputs:"
      1.) USB Webcam.
      2.) A directory of images (files in directory matching 'img_glob').
      3.) A video file, such as an .mp4 or .avi file.
    """

    def __init__(self, basedir, camid, h_ratio, w_ratio):
        self.cap = []
        self.video_file = False
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
        self.i = 0
        self.num_frames = 0
        # If the "basedir" string is the word camera, then use a webcam.
        if basedir == "camera/" or basedir == "camera":
            print('==> Processing Webcam Input.')
            self.cap = cv2.VideoCapture(camid)
        else:
            # Try to open as a video.
            self.cap = cv2.VideoCapture(basedir)
            lastbit = basedir[-4:len(basedir)]
            if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
                raise IOError('Cannot open movie file')
            elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
                print('==> Processing Video Input.')
                self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_file = True

    def next_frame(self, subsample_rate):
        """ Return the next frame, and increment internal counter.
        Returns
           image: Next H x W image.
           status: True or False depending whether image was loaded.
        """
        if self.i == self.num_frames:
            return (None, None, 'max_len')
        ret, input_image = self.cap.read()
        if ret is False:
            return (None, None, False)

        input_image = cv2.resize(input_image,
                                 (int(input_image.shape[1] * subsample_rate),
                                  int(input_image.shape[0] * subsample_rate)))

        image_shape = input_image.shape[:2]
        patch_shape = np.asarray([int(input_image.shape[0] * self.h_ratio),
                                  int(input_image.shape[1] * self.w_ratio)])

        # center patch
        center_image_lu_pt = (image_shape - patch_shape) // 2
        center_image = input_image[
                       center_image_lu_pt[0]:center_image_lu_pt[0] + patch_shape[0],
                       center_image_lu_pt[1]:center_image_lu_pt[1] + patch_shape[1]].copy()
        center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2GRAY)

        # left-up patch
        lu_image = input_image[5:patch_shape[0]+5, 5:patch_shape[1]+5].copy()
        lu_image = cv2.cvtColor(lu_image, cv2.COLOR_RGB2GRAY)

        # right-up patch
        ru_image_lu_pt = np.array([5, image_shape[1] - patch_shape[1]-5])
        ru_image = input_image[
                   ru_image_lu_pt[0]:ru_image_lu_pt[0] + patch_shape[0],
                   ru_image_lu_pt[1]:ru_image_lu_pt[1] + patch_shape[1]].copy()
        ru_image = cv2.cvtColor(ru_image, cv2.COLOR_RGB2GRAY)

        # left-down patch
        ld_image_lu_pt = np.array([image_shape[0] - patch_shape[0]-5, 5])
        ld_image = input_image[
                   ld_image_lu_pt[0]:ld_image_lu_pt[0] + patch_shape[0],
                   ld_image_lu_pt[1]:ld_image_lu_pt[1] + patch_shape[1]].copy()
        ld_image = cv2.cvtColor(ld_image, cv2.COLOR_RGB2GRAY)

        # right-down patch
        rd_image_lu_pt = np.array([image_shape[0] - patch_shape[0]-5, image_shape[1] - patch_shape[1]-5])
        rd_image = input_image[
                   rd_image_lu_pt[0]:rd_image_lu_pt[0] + patch_shape[0],
                   rd_image_lu_pt[1]:rd_image_lu_pt[1] + patch_shape[1]].copy()
        rd_image = cv2.cvtColor(rd_image, cv2.COLOR_RGB2GRAY)

        patches = np.array([center_image, lu_image, ru_image, ld_image, rd_image])
        patches = patches.astype('float32') / 255.0
        # Increment internal counter.
        self.i = self.i + 1
        return (input_image, patches, True)

if __name__  == "__main__":
    # # 使用示例
    # pdf_path = './tmp_data/dropout.pdf'  # 替换为你的 PDF 文件路径
    # output_folder = './tmp_data/image'  # 替换为你想保存图像的文件夹路径
    # # pdf_to_images(pdf_path, output_folder)
    #
    # # 打开PDF文件
    # doc = fitz.open(pdf_path)
    #
    # # 遍历每一页并转换为图片
    # for page_number in range(len(doc)):
    #     page = doc.load_page(page_number)  # 获取页面对象
    #     pix = page.get_pixmap()  # 获取页面图像对象
    #     pix.save(f'{output_folder}/{page_number}.png')  # 保存为PNG格式


    # hash_method = cv2.img_hash.PHash_create()
    # print(hash_method)

    vs = VideoStreamer("camera", 0, 0.6, 0.6)

    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 0 表示默认的摄像头

    # 检查摄像头是否打开
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        # 捕获帧
        ret, frame = cap.read()

        # 检查帧是否成功捕获
        if not ret:
            print("Error: Could not read frame.")
            break

        # 显示捕获的帧
        cv2.imshow('Camera Feed', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源
    cap.release()
    cv2.destroyAllWindows()