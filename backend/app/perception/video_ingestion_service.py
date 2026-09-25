"""
ChronoEye Infinity - Phase 2: Video Ingestion Service
Manages real-time video stream ingestion, recorded file decoding, frame order preservation,
frame metadata generation, stream health state monitoring, and configurable frame sampling.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from app.perception.frame_source import (
    BaseFrameSource,
    VideoFrameSource,
    CameraFrameSource,
    SyntheticFrameSource,
    FrameMetadata,
    SourceType,
    CameraStreamStatus,
)
from app.perception.preprocessor import VideoPreprocessor, PreprocessorConfig


class StreamStats(BaseModel):
    """
    Real-time performance metrics for an active camera stream.
    """
    camera_id: str
    source_id: str
    source_type: SourceType
    status: CameraStreamStatus
    frames_processed: int = 0
    processing_fps: float = 0.0
    latest_frame_id: int = 0
    latest_timestamp: float = 0.0
    last_successful_frame_time: float = 0.0
    stale_frame_count: int = 0
    reconnection_attempts: int = 0


class VideoIngestionService:
    """
    Master service coordinating active frame ingestion streams across CCTV cameras and recorded video files.
    """

    def __init__(self):
        self.sources: Dict[str, BaseFrameSource] = {}
        self.preprocessors: Dict[str, VideoPreprocessor] = {}
        self.stats: Dict[str, StreamStats] = {}
        self._latest_frames: Dict[str, Tuple[Any, FrameMetadata]] = {}

    def register_video_source(
        self,
        camera_id: str,
        video_path: str,
        source_id: Optional[str] = None,
        target_fps: Optional[float] = 10.0,
        preprocessor_config: Optional[PreprocessorConfig] = None,
    ) -> StreamStats:
        """
        Registers and initializes a Recorded Video file source (MP4, AVI, MOV).
        """
        src_id = source_id or f"SRC_VID_{camera_id}"
        source = VideoFrameSource(
            video_path=video_path,
            camera_id=camera_id,
            source_id=src_id,
            target_fps=target_fps,
        )
        self.sources[camera_id] = source
        self.preprocessors[camera_id] = VideoPreprocessor(preprocessor_config or PreprocessorConfig())

        stat = StreamStats(
            camera_id=camera_id,
            source_id=src_id,
            source_type=SourceType.RECORDED_VIDEO,
            status=source.status,
        )
        self.stats[camera_id] = stat
        return stat

    def register_camera_stream(
        self,
        camera_id: str,
        stream_url_or_device: Any = 0,
        source_id: Optional[str] = None,
        target_fps: float = 10.0,
        preprocessor_config: Optional[PreprocessorConfig] = None,
    ) -> StreamStats:
        """
        Registers and initializes a Live Camera stream (RTSP, IP Cam, Webcam).
        """
        src_id = source_id or f"SRC_CAM_{camera_id}"
        source = CameraFrameSource(
            stream_url_or_device=stream_url_or_device,
            camera_id=camera_id,
            source_id=src_id,
            target_fps=target_fps,
        )
        self.sources[camera_id] = source
        self.preprocessors[camera_id] = VideoPreprocessor(preprocessor_config or PreprocessorConfig())

        stat = StreamStats(
            camera_id=camera_id,
            source_id=src_id,
            source_type=SourceType.LIVE_STREAM,
            status=source.status,
        )
        self.stats[camera_id] = stat
        return stat

    def register_synthetic_source(
        self,
        camera_id: str = "CAM_SYNTHETIC",
        source_id: Optional[str] = None,
        total_frames: int = 1000,
        fps: float = 30.0,
    ) -> StreamStats:
        """
        Registers a synthetic frame generator fallback source.
        """
        src_id = source_id or f"SRC_SYN_{camera_id}"
        source = SyntheticFrameSource(
            camera_id=camera_id,
            source_id=src_id,
            total_frames=total_frames,
            fps=fps,
        )
        self.sources[camera_id] = source
        self.preprocessors[camera_id] = VideoPreprocessor(PreprocessorConfig())

        stat = StreamStats(
            camera_id=camera_id,
            source_id=src_id,
            source_type=SourceType.SIMULATION,
            status=source.status,
        )
        self.stats[camera_id] = stat
        return stat

    def get_next_frame(
        self, camera_id: str
    ) -> Tuple[bool, Optional[Any], Optional[FrameMetadata], Dict[str, Any]]:
        """
        Reads and preprocesses the next frame from a camera stream.
        """
        if camera_id not in self.sources:
            return False, None, None, {"error": f"Camera {camera_id} not registered"}

        source = self.sources[camera_id]
        preprocessor = self.preprocessors[camera_id]
        stat = self.stats[camera_id]

        t0 = time.time()
        success, raw_frame, meta = source.read_frame()

        stat.status = source.status

        if not success or raw_frame is None or meta is None:
            stat.stale_frame_count += 1
            return False, None, None, {"status": stat.status.value}

        # Apply preprocessing (resize, contrast, normalization, ROI)
        processed_frame, prep_info = preprocessor.process_frame(
            raw_frame, meta.width, meta.height
        )

        t_elapsed = time.time() - t0
        stat.frames_processed += 1
        stat.latest_frame_id = meta.frame_id
        stat.latest_timestamp = meta.timestamp
        stat.last_successful_frame_time = time.time()
        stat.processing_fps = round(1.0 / max(0.001, t_elapsed), 2)

        self._latest_frames[camera_id] = (processed_frame, meta)

        return True, processed_frame, meta, prep_info

    def get_stream_status(self, camera_id: str) -> Optional[StreamStats]:
        """
        Returns real-time stream status for a camera.
        """
        return self.stats.get(camera_id)

    def get_all_stream_statuses(self) -> List[StreamStats]:
        """
        Returns stream statuses for all registered cameras.
        """
        return list(self.stats.values())

    def release_camera(self, camera_id: str):
        """
        Releases video resources for a specific camera.
        """
        if camera_id in self.sources:
            self.sources[camera_id].release()
            if camera_id in self.stats:
                self.stats[camera_id].status = CameraStreamStatus.OFFLINE
