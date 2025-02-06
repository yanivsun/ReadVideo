import os
import torch
import cv2
from torchvision import transforms

def extract_and_save_frames(video_path, save_folder, interval=30):
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = []

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_name = f"frame_{frame_count:04d}.jpg"
            frame_path = os.path.join(save_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_frames.append(frame_path)
        frame_count += 1

    cap.release()

    if not saved_frames:
        raise ValueError("No frames were extracted. Check the video path or interval.")

    return saved_frames


def preprocess_frames(frames):
    '''
    将图像进行预处理，转成tensor（对大模型来讲或许不需要，对CLIP可能需要）
    :param frames:
    :return:
    '''
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    preprocessed_frames = []
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read frame: {frame_path}")

        # Apply the transformations to the frame and convert to tensor
        preprocessed_frame = transform(frame[:, :, ::-1])  # Convert from BGR to RGB
        preprocessed_frames.append(preprocessed_frame)

    if not preprocessed_frames:
        raise ValueError("No frames were preprocessed. Check the input frames.")

    # Stack all preprocessed frames into a single batch (4D tensor)
    frames_tensor = torch.stack(preprocessed_frames)

    # Ensure the tensor is of shape (num_frames, 3, 224, 224)
    print(f"Preprocessed frames shape: {frames_tensor.shape}")  # This should print something like: [54, 3, 224, 224]

    return frames_tensor