import torch
from doctr.models import ocr_predictor
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import os

device = (
    "mps"#APPLE DEVICE
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
model = ocr_predictor(
    "db_mobilenet_v3_large", "crnn_mobilenet_v3_large", pretrained=True
)
model.to(device)

# 图像预处理
def process_image(image_path:str):
    '''
    preprocesses a single image.

    :param image_path:
           The path to the image file.
    :return:
           np.array: The preprocessed image tensor.
    '''
    image = Image.open(image_path)
    image = np.array(image)
    if image.shape[-1] == 4: # [W,H,C] C -> RGBA
        image = image[:,:,:3]
    if len(image.shape) == 2:#灰度图
        image = np.stack([image] * 3, axis=-1)
    return image

def process_page(page, OCR_threshold):
    """
        Processes a single OCR page and extracts text based on a confidence threshold.

        Args:
            page: An OCR page object containing blocks, lines, and words.
            OCR_threshold (float): The confidence threshold for including words in the extracted text.

        Returns:
            str: The extracted text if it meets the criteria, otherwise None.
    """
    blocks = page.blocks
    try:
        text = " ".join(
            word.value
            for block in blocks
            for line in block.lines
            for word in line.words
            if word.confidence > OCR_threshold
        )
    except:
        text = None
    if text == "" or (
            text is not None
            and (not any(char.isalpha() for char in text) or len(text) < 3)
            or all(len(word) == 1 for word in text.split() if word.isalpha())
    ):
        text = None
    return text

def apply_OCR(image_paths, OCR_threshold=0.5):
    """
        Applies Optical Character Recognition (OCR) on images and returns the recognized text.

        Args:
            image_paths (list of str): The paths to the image files.
            OCR_threshold (float, optional): The confidence threshold for the OCR detection. Defaults to 0.5.

        Returns:
            list of str or None: The recognized text for each image if any text is detected, otherwise None.
    """
    global model
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, image_paths))
    results = model(images)

    def process_page_wrapper(page):
        return process_page(page, OCR_threshold)
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(process_page_wrapper, results.pages))

    # delete model and free up memory
    del model
    torch.cuda.empty_cache()
    return texts

if __name__ == "__main__":
    directory_path = '../tmp_data/image'
    files = []
    for filename in os.listdir(directory_path):
        filename_tmp = f"{directory_path}/{filename}"
        files.append(filename_tmp)
    print(files)
    aa = apply_OCR(files)
    print(aa)