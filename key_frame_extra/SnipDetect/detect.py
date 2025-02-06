import logging
import threading
import time
import cv2
import glob
import numpy as np
import argparse
import os
import requests
import json
import pika
import signal
import sys
from datetime import datetime, timedelta, timezone
from stream_precess import HLSStreamProcessor

last_successful_read = time.time()
stop_flag = False  # Used to properly stop threads on exit

def health_check(interval):
    """Health check that runs in a separate thread."""
    while not stop_flag:
        time.sleep(interval)
        elapsed_time = time.time() - last_successful_read

        if elapsed_time > interval * 2:
            print(f"[ERROR] No frame processed for {elapsed_time:.1f} seconds! Something might be stuck.")
            sys.exit(1)
        else:
            print(f"Ping")

def extract_keyframes_from_video(snippet_video_path, output_dir):
    """
    Extracts the first frame, the last frame, and a frame approx every 1s based on video FPS.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(snippet_video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open snippet video: {snippet_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0  # fallback if unknown

    # One frame per second approx
    frame_interval = int(round(fps))
    if frame_interval < 1:
        frame_interval = 1

    # Extract first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(output_dir, "frame_000001.jpg"), frame)

    # Extract intermediate frames
    current_frame = frame_interval
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{current_frame:06d}.jpg"), frame)
        current_frame += frame_interval

    # Extract last frame if not already done
    if total_frames > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(output_dir, f"frame_{total_frames:06d}.jpg"), frame)

    cap.release()
    print(f"Extracted keyframes to {output_dir}")

def load_image_as_snippet(image_path, hash_method, match_threshold):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    h = hash_method.compute(img)
    return [h], [image_path]  # One hash, one "frame path" as a placeholder

def load_or_extract_snippet(snippet_path, hash_method, match_threshold):
    if os.path.isfile(snippet_path):
        ext = os.path.splitext(snippet_path)[1].lower()
        if ext in [".jpg", ".png", ".jpeg", ".bmp"]:
            # It's a single image
            return load_image_as_snippet(snippet_path, hash_method, match_threshold)
        else:
            # It's a video file, proceed as usual
            base_name = os.path.splitext(os.path.basename(snippet_path))[0]
            frames_dir = base_name + "_frames"
            if not os.path.exists(frames_dir) or len(glob.glob(frames_dir + "/*.jpg")) == 0:
                extract_keyframes_from_video(snippet_path, frames_dir)
            return load_snippet_keyframes(frames_dir, hash_method, match_threshold)
    else:
        # It's assumed to be a directory of frames
        return load_snippet_keyframes(snippet_path, hash_method, match_threshold)

def load_snippet_keyframes(snippet_frames_dir, hash_method, match_threshold):
    """Load and hash all keyframes from a frames directory."""
    frame_paths = sorted(glob.glob(f"{snippet_frames_dir}/*.jpg"))
    if not frame_paths:
        raise ValueError(f"No keyframe images found in {snippet_frames_dir}.")

    known_hashes = []
    for fpath in frame_paths:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h = hash_method.compute(img)
        known_hashes.append(h)
    if not known_hashes:
        raise ValueError(f"No valid keyframes loaded from {snippet_frames_dir}")
    return known_hashes, frame_paths


def frame_matches_keyframe(frame, keyframe_hash, hash_method, threshold):
    frame_hash = hash_method.compute(frame)
    xor_val = np.bitwise_xor(frame_hash, keyframe_hash)
    hamming_distance = np.count_nonzero(xor_val)
    return hamming_distance < threshold


def reset_snippet_state(snippet):
    snippet['current_keyframe_index'] = 0
    snippet['start_time_for_current_keyframe'] = None
    snippet['start_snippet_time'] = None


def check_and_update_snippet(snippet, frame, current_timestamp, current_time, hash_method,
                             match_threshold, time_window, fps):
    # 1) Check for cooldown (avoid repeated detections too close in time)
    if snippet['last_detection_timestamp'] is not None:
        time_since_last = (current_timestamp - snippet['last_detection_timestamp']).total_seconds()
        if time_since_last < snippet['cooldown']:
            # Still within cooldown, skip checks
            return False, None, None, None, None

    # 2) Single-image snippet logic
    if len(snippet['known_hashes']) == 1:
        # Compare the current frame with the single known hash
        if frame_matches_keyframe(frame, snippet['known_hashes'][0], hash_method, match_threshold):
            snippet['consecutive_matches'] += 1
        else:
            snippet['consecutive_matches'] = 0

        # If we've matched enough consecutive frames to exceed min duration
        consecutive_match_time = snippet['consecutive_matches'] / fps
        if consecutive_match_time >= snippet['image_min_duration']:
            # Mark detection
            snippet['last_detection_timestamp'] = current_timestamp
            snippet['consecutive_matches'] = 0

            # Return detection info (start_time and end_time can be the same for an image, or a small offset)
            start_time = current_time - consecutive_match_time
            end_time   = current_time  # or current_time for a “point in time”
            start_timestamp = current_timestamp - timedelta(seconds=consecutive_match_time)
            end_timestamp   = current_timestamp

            return True, start_time, end_time, start_timestamp, end_timestamp
        else:
            return False, None, None, None, None

    # 3) Multi-keyframe snippet logic (the original code)
    #    (You can keep the existing approach or adapt it slightly.)
    #    ...
    #    Original snippet matching code below:
    # -------------------------------------------------------------------
    # If we've started matching but haven't found the next keyframe in time, reset
    if snippet['current_keyframe_index'] > 0 and snippet['start_time_for_current_keyframe'] is not None:
        if (current_timestamp - snippet['start_time_for_current_keyframe']).total_seconds() > time_window:
            reset_snippet_state(snippet)

    # Check frame against the current needed keyframe
    if frame_matches_keyframe(frame, snippet['known_hashes'][snippet['current_keyframe_index']],
                              hash_method, match_threshold):
        if snippet['current_keyframe_index'] == 0:
            snippet['start_snippet_timestamp'] = current_timestamp
            snippet['start_snippet_time'] = current_time

        snippet['current_keyframe_index'] += 1
        snippet['start_time_for_current_keyframe'] = current_timestamp

        # All keyframes matched
        if snippet['current_keyframe_index'] == len(snippet['known_hashes']):
            start_timestamp = snippet['start_snippet_timestamp']
            start_time      = snippet['start_snippet_time']
            # This is a rough calc of snippet end time
            end_time = start_time + int(snippet['frame_paths'][-1][-10:-4]) / fps
            end_timestamp = start_timestamp + timedelta(seconds=int(snippet['frame_paths'][-1][-10:-4]) / fps)

            snippet['last_detection_timestamp'] = current_timestamp
            reset_snippet_state(snippet)
            return True, start_time, end_time, start_timestamp, end_timestamp

    return False, None, None, None, None



def notify_server(url, clip_name, start_time, end_time):
    data = {
        "clip_name": clip_name,
        "start_time": start_time,
        "end_time": end_time
    }
    try:
        response = requests.post(url, json=data)
        print(f"Notification sent to {url}, response status: {response.status_code}")
    except Exception as e:
        print(f"Failed to notify server at {url}: {e}")

def notify_amqp_server(channel, clip_name, start_time, end_time, queue_name):
    data = {
        "clip_name": clip_name,
        "start_time": start_time,
        "end_time": end_time
    }
    channel.queue_declare(queue=queue_name)
    channel.basic_publish(exchange='',
                      routing_key=queue_name,
                      body=json.dumps(data))


def main():
    global last_successful_read, stop_flag
    parser = argparse.ArgumentParser(
        description="Detect multiple known snippets in a video or stream, extracting keyframes if needed.")
    parser.add_argument("--source", required=True, help="Video file path or stream URL (RTMP/HLS).")
    parser.add_argument("--clips", nargs='+', required=True,
                        help="Paths to snippet directories or videos. If video is given, keyframes are extracted automatically.")
    parser.add_argument("--match_threshold", type=int, default=5, help="Hamming distance threshold for frame matching.")
    parser.add_argument("--image_min_duration", type=float, default=1.0,
                        help="Minimum duration (in seconds) that a single image must be visible to be considered 'detected'.")
    parser.add_argument("--detection_cooldown", type=float, default=10.0,
                        help="Minimum time in seconds between repeated detections of the same snippet.")

    parser.add_argument("--time_window", type=float, default=3.0,
                        help="Time window (in seconds) to find next keyframe.")
    parser.add_argument("--hash_method", type=str, default="phash", choices=["phash", "average", "marr", "radial"],
                        help="Image hash method.")
    parser.add_argument("--notify_url", type=str, help="If provided, POST detection results to this URL.")
    parser.add_argument("--display", action="store_true", help="Display stream frame")
    parser.add_argument("--amqp_url", default=None, type=str,
                        help="AMQP URL in the format amqp://username:password@host")
    parser.add_argument("--health_check_interval", type=int, default=15,
                        help="Print a health check message every X seconds (set 0 to disable).")
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")

    args = parser.parse_args()

    if args.health_check_interval > 0:
        threading.Thread(target=health_check, args=(args.health_check_interval,), daemon=True).start()

    # Select hash method
    if args.hash_method == "phash":
        hash_method = cv2.img_hash.PHash_create()
    elif args.hash_method == "average":
        hash_method = cv2.img_hash.AverageHash_create()
    elif args.hash_method == "marr":
        hash_method = cv2.img_hash.MarrHildrethHash_create()
    elif args.hash_method == "radial":
        hash_method = cv2.img_hash.RadialVarianceHash_create()

    if args.amqp_url:
        connection = pika.BlockingConnection(pika.URLParameters(args.amqp_url))  # TODO add parameters for connection
        channel = connection.channel()
    # Load all snippets
    snippets = {}
    for clip_path in args.clips:
        known_hashes, frame_paths = load_or_extract_snippet(clip_path, hash_method, args.match_threshold)
        snippets[clip_path] = {
            'known_hashes': known_hashes,
            'current_keyframe_index': 0,
            'start_time_for_current_keyframe': None,
            'start_snippet_time': None,
            'frame_paths': frame_paths,
            'last_detection_timestamp': None,  # For cooldown
            'consecutive_matches': 0,  # For single images
            'image_min_duration': args.image_min_duration,
            'cooldown': args.detection_cooldown
        }
    is_hls = False
    if args.source.endswith((".m3u8")):
        cap = HLSStreamProcessor(args.source, max_queue_size=20)
        is_hls = True
    else:
        cap = cv2.VideoCapture(args.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not cap.isOpened():
            raise IOError(f"Could not open video source {args.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback if fps not available

    try:

        frame_nb = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video or no stream data
                break
            last_successful_read = time.time()  # Update last good read time

            # Update current timestamp
            current_timestamp = datetime.now()
            if is_hls:

                fps = frame["fps"] if "fps" in frame else 30

                frame_number_in_chunk = frame["frame_number_in_chunk"]
                chunk_name = frame["chunk_name"]
                program_date_time = frame["program_date_time"]
                frame = frame["frame"]
                current_timestamp = datetime.fromisoformat(str(program_date_time)) + timedelta(
                    seconds=frame_number_in_chunk / fps)
                if args.verbose:
                    logging.info(f"Chunk: {chunk_name}, Frame#: {frame_number_in_chunk}, "
                                 f"FPS: {fps}, Timestamp: {current_timestamp}, PDT: {program_date_time}")
            current_time = frame_nb / fps

            # Check each snippet
            for snippet_name, snippet in snippets.items():
                detected, start_time, end_time, start_timestamp, end_timestamp = check_and_update_snippet(
                    snippet, frame, current_timestamp, current_time, hash_method, args.match_threshold,
                    args.time_window, fps
                )
                if detected:
                    print(
                        f"[{snippet_name}] Detected snippet! Start: {start_time:.2f}s, End: {end_time:.2f}s, Timestamp: {start_timestamp}, End Timestamp: {end_timestamp}")
                    if args.notify_url:
                        notify_server(args.notify_url, snippet_name, start_time, end_time)
                    if args.amqp_url:
                        notify_amqp_server(channel, snippet_name, start_time, end_time, 'snippets')

            # display stream frame
            if args.display:
                cv2.imshow("stream", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            frame_nb += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # print message of end
        print("End of video")


if __name__ == "__main__":
    main()

    '''
      python detect_clips.py \
      --source input_video.mp4 \
      --clips snippet1.mp4 snippet2_frames snippet3.jpg \
      --match_threshold 5 \
      --time_window 3.0 \
      --image_min_duration 1.5 \
      --detection_cooldown 10.0 \
      --hash_method phash \
      --notify_url http://yourserver.com/api/detect \
      --health_check_interval 15 \
      --display \
      --verbose
      
      
      python detect_clips.py \
      --source input_video.mp4 \
      --clips snippet1.mp4 snippet2_frames snippet3.jpg \
      --match_threshold 5 \
      --time_window 3.0 \
      --hash_method phash \
      --notify_url http://localhost:5000/detection-event \
      --health_check_interval 10 \
      --display \
      --verbose
    '''