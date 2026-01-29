import uuid
import subprocess
import os
import requests
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field

# ---------------- CONFIG ----------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TEMP_DIR = Path("temp_inputs")  # New folder for downloading inputs
TEMP_DIR.mkdir(exist_ok=True)

# ---------------------------------------

app = FastAPI(title="LipSync Worker")

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ---------------- MODELS ----------------

class LipSyncRequest(BaseModel):
    video_url: HttpUrl
    audio_url: HttpUrl
    guidance_scale: float = Field(1.5, ge=0.1, le=5.0)
    inference_steps: int = Field(25, ge=10, le=60)

class LipSyncResponse(BaseModel):
    job_id: str
    video_url: str

# ---------------- UTILS ----------------

def download_file(url: str, save_path: Path):
    """Helper to download files from URLs to local disk"""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {str(e)}")

# ---------------- INFERENCE ----------------

def run_lipsync(local_video_path, local_audio_path, output_path, steps, scale):
    cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--inference_steps", str(steps),
        "--guidance_scale", str(scale),
        "--video_path", str(local_video_path),     # <-- NOW USING LOCAL PATH
        "--audio_path", str(local_audio_path),     # <-- NOW USING LOCAL PATH
        "--video_out_path", str(output_path)
    ]

    print(f"Running command: {' '.join(cmd)}") # Debug print

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Print logs to terminal for debugging
    print("STDOUT:", process.stdout)
    
    if process.returncode != 0:
        print("STDERR:", process.stderr) # This will show you the exact error
        raise RuntimeError(f"Inference Failed: {process.stderr}")

# ---------------- ROUTES ----------------

@app.get('/')
def root():
    return {"msg": "LatentSync API is Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate", response_model=LipSyncResponse)
def generate(req: LipSyncRequest):
    job_id = str(uuid.uuid4())
    
    # 1. Define paths for temporary storage
    local_video_path = TEMP_DIR / f"{job_id}_video.mp4"
    local_audio_path = TEMP_DIR / f"{job_id}_audio.wav"
    final_output_path = OUTPUT_DIR / f"{job_id}.mp4"

    try:
        print(f"Downloading inputs for Job {job_id}...")
        # 2. DOWNLOAD the files from the URLs
        download_file(str(req.video_url), local_video_path)
        download_file(str(req.audio_url), local_audio_path)

        # 3. Run Inference using the LOCAL files
        print(f"Starting Inference for Job {job_id}...")
        run_lipsync(
            local_video_path,
            local_audio_path,
            final_output_path,
            req.inference_steps,
            req.guidance_scale
        )

        return {
            "job_id": job_id,
            "video_url": f"/outputs/{job_id}.mp4"
        }

    except Exception as e:
        print(f"Job Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # 4. Cleanup temp input files to save space
        if local_video_path.exists():
            os.remove(local_video_path)
        if local_audio_path.exists():
            os.remove(local_audio_path)