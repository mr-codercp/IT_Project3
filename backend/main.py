from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.video_processor import get_video_info, sample_frames
from backend.detector import detect_ai_probability

import os
import shutil
import uuid


app = FastAPI(title="AI Video Detector")


# -----------------------------
# CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Upload settings
# -----------------------------

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


ALLOWED_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


# -----------------------------
# Home
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "AI Video Detector backend is running"
    }


# -----------------------------
# Health check
# -----------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


# -----------------------------
# Analyze video
# -----------------------------

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):

    # Check extension
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    # Safe temporary filename
    filename = f"{uuid.uuid4()}{extension}"

    file_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    try:

        # -----------------------------
        # Save uploaded video
        # -----------------------------

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # -----------------------------
        # Get video information
        # -----------------------------

        video_info = get_video_info(
            file_path
        )

        # -----------------------------
        # Sample 20 frames
        # -----------------------------

        frames = sample_frames(
            file_path,
            sample_count=20
        )

        # -----------------------------
        # Run AI detector
        # -----------------------------

        frame_results = []

        for index, frame_data in enumerate(
            frames,
            start=1
        ):

            ai_probability = detect_ai_probability(
                frame_data["image"]
            )

            frame_results.append({
                "frame": index,

                "timestamp": round(
                    frame_data["timestamp"],
                    2
                ),

                "ai_probability": round(
                    ai_probability * 100,
                    2
                ),

                "prediction": (
                    "AI"
                    if ai_probability >= 0.65
                    else "Real"
                )
            })

        # -----------------------------
        # Calculate video probability
        # -----------------------------

        if not frame_results:
            raise ValueError(
                "No frames were successfully analyzed."
            )

        ai_probability = sum(
            item["ai_probability"]
            for item in frame_results
        ) / len(frame_results)

        real_probability = (
            100 - ai_probability
        )

        # -----------------------------
        # Final classification
        # -----------------------------

        if ai_probability >= 65:

            classification = (
                "Likely AI-generated"
            )

        else:

            classification = (
                "Probably Real"
            )

        # -----------------------------
        # Return result
        # -----------------------------

        return {

            "success": True,

            "filename": file.filename,

            "frames_analyzed": len(frames),

            "fps": round(
                video_info["fps"],
                2
            ),

            "frame_count": (
                video_info["frame_count"]
            ),

            "duration_seconds": round(
                video_info["duration"],
                2
            ),

            "ai_probability": round(
                ai_probability,
                2
            ),

            "real_probability": round(
                real_probability,
                2
            ),

            "classification": classification,

            "frame_results": frame_results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        await file.close()

        # Delete temporary video
        if os.path.exists(file_path):

            os.remove(file_path)