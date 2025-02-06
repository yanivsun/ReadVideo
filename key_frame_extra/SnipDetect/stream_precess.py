'''
https://github.com/travisseng/SnippetDetect/blob/main/hls_stream_processor.py
'''
import requests
import av
import time
import m3u8
from io import BytesIO
import threading
from queue import Queue, Full
import cv2
import logging

logging.basicConfig(level=logging.INFO)

class HLSStreamProcessor:
    def __init__(self, hls_url, max_queue_size=0):
        """
        :param hls_url: The URL of the HLS media playlist (or master, if you handle variant logic).
        :param max_queue_size: Optional maximum size of the segment queue (0 = unlimited).
        """
        self.hls_url = hls_url
        self.etag = None
        self.last_modified = None
        self.last_processed_segment = None
        self.segment_queue = Queue(maxsize=max_queue_size)
        self.playlist_base_uri = None
        self.stop_event = threading.Event()

        # For CMAF init segment caching
        self.cached_init_data = None
        self.cached_init_url = None

        # Internal generator reference for read() usage
        self._frame_generator_iter = None

        # Thread reference
        self._downloader_thread = None
        self.start()

    def fetch_playlist(self):
        headers = {}
        if self.etag:
            headers['If-None-Match'] = self.etag
        if self.last_modified:
            headers['If-Modified-Since'] = self.last_modified

        try:
            response = requests.get(self.hls_url, headers=headers)
        except requests.RequestException as e:
            logging.error(f"Error fetching playlist: {e}")
            return None
        if response.status_code == 200:
            self.etag = response.headers.get('ETag')
            self.last_modified = response.headers.get('Last-Modified')
            playlist = m3u8.loads(response.text)

            # If it's a master playlist, handle or warn
            if playlist.is_variant:
                logging.warning("Received a master playlist. You may need to select a specific variant.")

            # CMAF segment_map check
            if playlist.segment_map:
                base_uri = self.hls_url.rsplit('/', 1)[0] + '/'
                playlist.base_uri = base_uri
                # If there's an init segment
                if len(playlist.segment_map) > 0:
                    playlist.segment_map[0].base_uri = base_uri
                self.playlist_base_uri = base_uri

            return playlist

        elif response.status_code == 304:
            # Not modified
            return None
        else:
            raise Exception(f"Failed to fetch playlist: HTTP {response.status_code}")

    def download_url(self, url, max_retries=3, timeout=5):
        """
        Simple helper to download URL with retries.
        """
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 200:
                    return resp.content
                else:
                    logging.error(f"Non-200 status code ({resp.status_code}) for URL: {url}")
            except requests.RequestException as e:
                logging.error(f"Download error for {url} [Attempt {attempt + 1}]: {e}")
            time.sleep(0.5)

        return None

    def get_combined_segment(self, init_url, chunk_url):
        """
        For CMAF: combine init segment + chunk data. Cache init segment if possible.
        """
        try:
            # Check if we've cached the init or if URL changed
            if self.cached_init_data is None or self.cached_init_url != init_url:
                logging.info(f"Downloading init segment from {init_url}")
                init_data = self.download_url(init_url)
                if init_data is None:
                    return None
                self.cached_init_data = init_data
                self.cached_init_url = init_url

            # Download chunk
            chunk_data = self.download_url(chunk_url)
            if chunk_data is None:
                return None

            return self.cached_init_data + chunk_data
        except Exception as e:
            logging.error(f"Error downloading segment {chunk_url}: {e}")
            return None

    def download_segments(self):
        """
        Thread target to continually fetch the playlist, find new segments,
        and enqueue them for decoding, starting from the live edge.
        """
        while not self.stop_event.is_set():
            try:
                playlist = self.fetch_playlist()
                if not playlist:
                    logging.warning("Empty or invalid playlist.")
                    time.sleep(1)
                    continue

                # If CMAF init segment is needed:
                init_url = None
                if playlist.segment_map and len(playlist.segment_map) > 0:
                    init_url = playlist.segment_map[0].absolute_uri

                segments = playlist.segments
                if not segments:
                    logging.warning("No segments in playlist.")
                    time.sleep(1)
                    continue

                # Convert to a list of URIs for easier checks
                segment_uris = [seg.absolute_uri for seg in segments]

                # ------------------------------------------------------
                # STEP 1: If first run (last_processed_segment is None),
                #         jump straight to the last segment (live edge).
                #         So we skip all previous segments in the playlist.
                # ------------------------------------------------------
                if self.last_processed_segment is None:
                    # Jump to the last segment (the newest one)
                    self.last_processed_segment = segments[-1].absolute_uri
                    logging.info(f"First run: jumping to live edge: {self.last_processed_segment}")
                    # We won't process that last segment right now;
                    # next iteration we will see "last_processed_segment" in the playlist
                    # and then process *newer* chunks after it.
                    time.sleep(1)
                    continue

                # ------------------------------------------------------
                # STEP 2: Check if our last_processed_segment is still in
                #         the playlist or if the playlist rolled over.
                # ------------------------------------------------------
                if self.last_processed_segment not in segment_uris:
                    # The playlist rolled over or we missed some segments
                    # Skip all older segments and jump to the newest again
                    logging.info(
                        f"Last processed segment '{self.last_processed_segment}' not in playlist. "
                        f"Skipping to latest segment."
                    )
                    self.last_processed_segment = segments[-1].absolute_uri
                    time.sleep(1)
                    continue

                # ------------------------------------------------------
                # STEP 3: Find last_processed_segment in the playlist
                #         and collect segments that come after it.
                # ------------------------------------------------------
                new_segments = []
                found_old = False
                for seg in segments:
                    if not found_old:
                        # if we haven't seen last_processed_segment yet
                        if seg.absolute_uri == self.last_processed_segment:
                            found_old = True
                        # skip until we match the old segment
                        continue
                    else:
                        # Found the old segment, so anything after that is "new"
                        new_segments.append(seg)

                # ------------------------------------------------------
                # STEP 4: Download and enqueue new segments
                # ------------------------------------------------------
                for seg in new_segments:
                    chunk_url = seg.absolute_uri

                    # Make sure it's not already in the queue
                    if not any(chunk_url == item["chunk_url"] for item in self.segment_queue.queue):
                        if init_url:
                            combined_data = self.get_combined_segment(init_url, chunk_url)
                        else:
                            # TS-based HLS
                            combined_data = self.download_url(chunk_url)

                        if combined_data:
                            segment_info = {
                                "chunk_url": chunk_url,
                                "combined_data": combined_data,
                                "program_date_time": seg.program_date_time,
                                "duration": seg.duration
                            }
                            try:
                                self.segment_queue.put(segment_info, block=True, timeout=2)
                            except Full:
                                logging.error("Segment queue is full; dropping segment.")

                        # Update the last_processed_segment after
                        # we successfully enqueue this chunk.
                        self.last_processed_segment = chunk_url

                time.sleep(1)

            except Exception as e:
                logging.error(f"Error in downloader thread: {e}")
                # Small delay before retry
                time.sleep(2)

    def decode_with_av(self, combined_data, chunk_url, program_date_time):
        """
        Decode frames from the combined (init + chunk) or TS segment.
        Yields frame information dictionaries.
        """
        try:
            with av.open(BytesIO(combined_data)) as container:
                if not container.streams.video:
                    logging.error(f"No video stream found for {chunk_url}.")
                    return

                video_stream = container.streams.video[0]
                fps = float(video_stream.average_rate) if video_stream.average_rate else 0.0
                frame_count = 0

                for frame in container.decode(video=0):
                    ndarray = frame.to_ndarray(format="bgr24")
                    yield {
                        "frame": ndarray,
                        "fps": fps,
                        "frame_number_in_chunk": frame_count,
                        "chunk_name": chunk_url,
                        "program_date_time": program_date_time,
                    }
                    frame_count += 1

        except av.AVError as av_err:
            logging.error(f"PyAV error decoding segment {chunk_url}: {av_err}")
        except Exception as e:
            logging.error(f"Unexpected error decoding {chunk_url}: {e}")

    def frame_generator(self):
        """
        Generator that pops segments off the queue and decodes them.
        """
        while not self.stop_event.is_set():
            if not self.segment_queue.empty():
                segment_info = self.segment_queue.get()
                chunk_url = segment_info["chunk_url"]
                combined_data = segment_info["combined_data"]
                program_date_time = segment_info.get("program_date_time", None)

                for frame_info in self.decode_with_av(combined_data, chunk_url, program_date_time):
                    yield frame_info
            else:
                time.sleep(0.1)

    def start(self):
        """
        Start the downloader thread and prepare the internal frame generator for read().
        """
        if self._downloader_thread is None or not self._downloader_thread.is_alive():
            self._downloader_thread = threading.Thread(
                target=self.download_segments, daemon=True
            )
            self._downloader_thread.start()

        # Create an iterator from the frame generator for read().
        self._frame_generator_iter = self.frame_generator()

    def read(self):
        """
        Mimics cv2.VideoCapture.read() style:
          :return: (ret, frame_info)
            - ret is True if a frame is returned, False otherwise
            - frame_info is a dictionary containing at least {'frame': ndarray, ...}
              or None if no frame is available / we're stopped.

        It will block momentarily if no data is available yet, but not forever.
        """
        if self.stop_event.is_set() or self._frame_generator_iter is None:
            return False, None

        try:
            frame_info = next(self._frame_generator_iter)
            return True, frame_info
        except StopIteration:
            # Generator exhausted or stop_event triggered
            return False, None
        except Exception as e:
            logging.error(f"Error reading next frame: {e}")
            return False, None

    def release(self):
        """
        Signal the downloader thread and any active generator to stop.
        """
        self.stop_event.set()
        if self._downloader_thread is not None:
            self._downloader_thread.join(timeout=2)


if __name__ == "__main__":
    hls_url = "https://rec-ofr-ingest-01.yuzzitpro.com/ISP/469511/raw2/stream_360p/master.m3u8"
    processor = HLSStreamProcessor(hls_url, max_queue_size=10)

    # Start the background downloader + frame generator

    try:
        while True:
            # read() returns (ret, frame_info)
            ret, frame_info = processor.read()
            if not ret:
                # No frame available currently
                # Short sleep to avoid busy loop
                time.sleep(0.01)
                continue

            frame = frame_info["frame"]
            fps = frame_info["fps"] if "fps" in frame_info else 0
            display_ms = int(1000 / fps) if fps > 0 else 33

            frame_number_in_chunk = frame_info["frame_number_in_chunk"]
            chunk_name = frame_info["chunk_name"]
            program_date_time = frame_info["program_date_time"]

            logging.info(f"Chunk: {chunk_name}, Frame#: {frame_number_in_chunk}, "
                         f"FPS: {fps}, PDT: {program_date_time}")

            cv2.imshow("HLS Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received; stopping...")
    finally:
        processor.release()
        cv2.destroyAllWindows()