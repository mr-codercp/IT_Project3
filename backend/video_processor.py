import cv2
import numpy as np


def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError("Could not read video information.")

    duration = frame_count / fps

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
    }


def sample_frames(video_path: str, sample_count: int = 20):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames.")

    sample_count = min(sample_count, total_frames)

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sample_count,
        dtype=int
    )

    sampled_frames = []

    for frame_index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        success, frame = cap.read()

        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fps = cap.get(cv2.CAP_PROP_FPS)

        timestamp = (
            frame_index / fps
            if fps > 0
            else 0
        )

        sampled_frames.append({
            "frame_number": int(frame_index),
            "timestamp": float(timestamp),
            "image": frame_rgb
        })

    cap.release()

    if not sampled_frames:
        raise ValueError("Could not extract any frames.")

    return sampled_frames