"""
ChronoEye Infinity - Phase 2: Frame Source & Video Ingestion Engine
Provides modular frame input interfaces for recorded video files (MP4, AVI, MOV),
live camera streams (RTSP, IP Camera, CCTV, Webcam), and synthetic frame generators.
Enforces real-world temporal frame metadata, configurable frame sampling,
and stream health state tracking.
"""

from abc import ABC, abstractmethod
from enum import Enum
import time
from typing import Dict, Generator, Optional, Tuple, Any
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    RECORDED_VIDEO = "RECORDED_VIDEO"
    LIVE_STREAM = "LIVE_STREAM"
    SIMULATION = "SIMULATION"


class CameraStreamStatus(str, Enum):
    LIVE = "LIVE"
    CONNECTING = "CONNECTING"
    RECONNECTING = "RECONNECTING"
    OFFLINE = "OFFLINE"
    STALE = "STALE"
    ERROR = "ERROR"


class FrameMetadata(BaseModel):
    """
    Standard metadata associated with every video frame ingested into ChronoEye.
    """
    frame_id: int
    source_id: str
    camera_id: str
    source_type: SourceType
    timestamp: float
    timestamp_iso: str
    frame_number: int
    width: int
    height: int
    fps: float
    original_fps: float
    sample_interval: int = 1
    stream_status: CameraStreamStatus = CameraStreamStatus.LIVE


class VideoConfig(BaseModel):
    """
    Configuration schema for Real Video / Live Camera ingestion (Stages 1-3).
    """
    source: Any = "0"  # File path, webcam index (0), or RTSP URL
    camera_id: str = "CAM_001"
    source_id: Optional[str] = None
    frame_skip: int = 0  # 0 means process every frame, 1 means skip 1 frame (process 1 of 2), etc.
    max_fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    device: str = "cpu"


import sys
import dis


class FlexibleFrameResult(tuple):
    """
    Standardize the 3-value frame output interface.
    """
    def __new__(cls, success: bool, frame: Any, meta: Optional[FrameMetadata]):
        return super().__new__(cls, (success, frame, meta))


class BaseFrameSource(ABC):
    """
    Abstract base class for ChronoEye frame sources.
    """

    def __init__(self, camera_id: str, source_id: Optional[str] = None):
        self.camera_id = camera_id
        self.source_id = source_id or camera_id
        self.frame_id = 0
        self.status = CameraStreamStatus.CONNECTING
        self.source_type = SourceType.SIMULATION
        self.error_message: Optional[str] = None

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[Any], Optional[FrameMetadata]]:
        """
        Reads next frame.
        Returns: (success, frame_data, FrameMetadata)
        """
        pass

    @abstractmethod
    def release(self):
        """Releases hardware/file stream resources."""
        pass


class VideoFrameSource(BaseFrameSource):
    """
    Frame source for recorded video files (MP4, AVI, MOV, MKV).
    Maintains original video temporal indexing and configurable frame skipping.
    """

    def __init__(
        self,
        video_path: str,
        camera_id: str = "CAM_FILE",
        source_id: str = "SRC_FILE_01",
        target_fps: Optional[float] = None,
        frame_skip: int = 0,
    ):
        super().__init__(camera_id, source_id)
        self.video_path = video_path
        self.source_type = SourceType.RECORDED_VIDEO
        self.original_fps = 30.0
        self.target_fps = target_fps
        self.frame_skip = max(0, frame_skip)
        self.sample_interval = self.frame_skip + 1
        self.width = 1920
        self.height = 1080
        self.cap = None
        self._is_open = False
        self.total_frames_in_file = 0
        self._init_cap()

    def _init_cap(self):
        import os
        if not os.path.exists(self.video_path):
            self._is_open = False
            self.status = CameraStreamStatus.ERROR
            self.error_message = f"Video file not found: {self.video_path}"
            return

        try:
            import cv2
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self._is_open = True
                self.status = CameraStreamStatus.LIVE
                self.error_message = None
                fps_det = self.cap.get(cv2.CAP_PROP_FPS)
                if fps_det > 0:
                    self.original_fps = fps_det
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_f = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if w > 0 and h > 0:
                    self.width = w
                    self.height = h
                if total_f > 0:
                    self.total_frames_in_file = total_f

                if self.target_fps and self.target_fps < self.original_fps:
                    self.sample_interval = max(self.sample_interval, int(round(self.original_fps / self.target_fps)))
            else:
                self._is_open = False
                self.status = CameraStreamStatus.ERROR
                self.error_message = f"Unable to open video source: {self.video_path}"
        except Exception as e:
            self._is_open = False
            self.status = CameraStreamStatus.ERROR
            self.error_message = f"Unable to open video source: {str(e)}"

    def read_frame(self) -> Tuple[bool, Optional[Any], Optional[FrameMetadata]]:
        if not self._is_open or self.cap is None:
            self.status = CameraStreamStatus.OFFLINE
            return FlexibleFrameResult(False, None, None)

        ret, frame = self.cap.read()
        if not ret:
            self.status = CameraStreamStatus.OFFLINE
            return FlexibleFrameResult(False, None, None)

        self.frame_id += 1

        # Handle frame skipping / sampling interval
        if (self.frame_id - 1) % self.sample_interval != 0:
            return FlexibleFrameResult(False, None, None)

        timestamp = self.frame_id / float(self.original_fps)
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(timestamp))

        effective_fps = self.original_fps / float(self.sample_interval)

        meta = FrameMetadata(
            frame_id=self.frame_id,
            source_id=self.source_id,
            camera_id=self.camera_id,
            source_type=self.source_type,
            timestamp=round(timestamp, 3),
            timestamp_iso=timestamp_iso,
            frame_number=self.frame_id,
            width=self.width,
            height=self.height,
            fps=round(effective_fps, 2),
            original_fps=round(self.original_fps, 2),
            sample_interval=self.sample_interval,
            stream_status=self.status,
        )

        return FlexibleFrameResult(True, frame, meta)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self._is_open = False
            self.status = CameraStreamStatus.OFFLINE


class CameraFrameSource(BaseFrameSource):
    """
    Frame source for live CCTV / RTSP / Webcam streams.
    Includes stale frame detection and auto-reconnection mechanics.
    """

    def __init__(
        self,
        stream_url_or_device: Any = 0,
        camera_id: str = "CAM_LIVE_01",
        source_id: str = "SRC_LIVE_01",
        target_fps: float = 10.0,
        frame_skip: int = 0,
    ):
        super().__init__(camera_id, source_id)
        self.stream_url_or_device = stream_url_or_device
        self.source_type = SourceType.LIVE_STREAM
        self.target_fps = target_fps
        self.frame_skip = max(0, frame_skip)
        self.sample_interval = self.frame_skip + 1
        self.original_fps = 30.0
        self.width = 1920
        self.height = 1080
        self.cap = None
        self._is_open = False
        self.last_read_time = 0.0
        self._init_cap()

    def _init_cap(self):
        try:
            import cv2
            self.status = CameraStreamStatus.CONNECTING
            self.cap = cv2.VideoCapture(self.stream_url_or_device)
            if self.cap.isOpened():
                self._is_open = True
                self.status = CameraStreamStatus.LIVE
                self.error_message = None
                fps_det = self.cap.get(cv2.CAP_PROP_FPS)
                if fps_det > 0:
                    self.original_fps = fps_det
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self.width = w
                    self.height = h
            else:
                self._is_open = False
                self.status = CameraStreamStatus.ERROR
                if isinstance(self.stream_url_or_device, int) or (isinstance(self.stream_url_or_device, str) and self.stream_url_or_device.isdigit()):
                    self.error_message = f"Webcam unavailable (device {self.stream_url_or_device})"
                elif str(self.stream_url_or_device).startswith("rtsp://"):
                    self.error_message = f"Unable to connect to RTSP stream: {self.stream_url_or_device}"
                else:
                    self.error_message = f"Unable to open video source: {self.stream_url_or_device}"
        except Exception as e:
            self._is_open = False
            self.status = CameraStreamStatus.ERROR
            self.error_message = f"Unable to open camera stream: {str(e)}"

    def read_frame(self) -> Tuple[bool, Optional[Any], Optional[FrameMetadata]]:
        now = time.time()
        if not self._is_open or self.cap is None:
            # Attempt auto-reconnect
            self.status = CameraStreamStatus.RECONNECTING
            self._init_cap()
            if not self._is_open:
                return FlexibleFrameResult(False, None, None)

        ret, frame = self.cap.read()
        if not ret:
            self.status = CameraStreamStatus.STALE if (now - self.last_read_time > 3.0) else CameraStreamStatus.ERROR
            return FlexibleFrameResult(False, None, None)

        self.last_read_time = now
        self.frame_id += 1
        self.status = CameraStreamStatus.LIVE

        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(now))

        meta = FrameMetadata(
            frame_id=self.frame_id,
            source_id=self.source_id,
            camera_id=self.camera_id,
            source_type=self.source_type,
            timestamp=round(now, 3),
            timestamp_iso=timestamp_iso,
            frame_number=self.frame_id,
            width=self.width,
            height=self.height,
            fps=round(self.target_fps, 2),
            original_fps=round(self.original_fps, 2),
            sample_interval=1,
            stream_status=self.status,
        )

        return FlexibleFrameResult(True, frame, meta)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self._is_open = False
            self.status = CameraStreamStatus.OFFLINE


class SyntheticFrameSource(BaseFrameSource):
    """
    Synthetic frame generator for environments without physical video files or cameras.
    Produces valid numpy frames or frame representations with rich traffic metadata.
    """

    def __init__(
        self,
        camera_id: str = "CAM_SYNTHETIC",
        source_id: str = "SRC_SYNTHETIC_01",
        total_frames: int = 1000,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
    ):
        super().__init__(camera_id, source_id)
        self.source_type = SourceType.SIMULATION
        self.total_frames = total_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.status = CameraStreamStatus.LIVE

    def read_frame(self) -> Tuple[bool, Optional[Any], Optional[FrameMetadata]]:
        if self.frame_id >= self.total_frames:
            self.status = CameraStreamStatus.OFFLINE
            return FlexibleFrameResult(False, None, None)

        self.frame_id += 1
        curr_time = time.time()
        timestamp = round(self.frame_id / self.fps, 3)
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(curr_time))

        frame = {"width": self.width, "height": self.height, "type": "synthetic"}

        meta = FrameMetadata(
            frame_id=self.frame_id,
            source_id=self.source_id,
            camera_id=self.camera_id,
            source_type=self.source_type,
            timestamp=timestamp,
            timestamp_iso=timestamp_iso,
            frame_number=self.frame_id,
            width=self.width,
            height=self.height,
            fps=self.fps,
            original_fps=self.fps,
            sample_interval=1,
            stream_status=self.status,
        )

        return FlexibleFrameResult(True, frame, meta)

    def release(self):
        self.status = CameraStreamStatus.OFFLINE

