"""Threaded webcam capture. Continuously grabs frames so the pipeline never blocks."""
import threading
import cv2
import numpy as np

from config import CameraConfig


class FrameGrabber:
    def __init__(self, camera_config: CameraConfig):
        self._cfg = camera_config
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._cfg.device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._cfg.device_id}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror for intuitive mapping
                with self._lock:
                    self._frame = frame

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            frame = self._frame.copy()
        return True, frame

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def release(self) -> None:
        self.stop()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
