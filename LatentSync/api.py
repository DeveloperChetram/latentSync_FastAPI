import uuid
import subprocess
import os
import requests
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, Field

# ---------------- CONFIG ----------------

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TEMP_DIR = Path("temp_inputs")
TEMP_DIR.mkdir(exist_ok=True)

# Replace this with your actual Lightning AI Public URL (Found in the browser address bar)
# Example: "https://8000-01kfnezv96p74g8kjjejgqqc0g.cloudspaces.litng.ai"
WORKER_PUBLIC_URL = os.environ.get("LIGHTNING_APP_URL", "http://localhost:8000")

# ---------------------------------------

app = FastAPI(title="LipSync Worker")

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ---------------- MODELS ----------------

class LipSyncRequest(BaseModel):
    video_url: HttpUrl
    audio_url: HttpUrl
    webhook_url: HttpUrl     # <--- NEW: Where to send the result
    main_job_id: str         # <--- NEW: ID from your main server
    guidance_scale: float = Field(1.5, ge=0.1, le=5.0)
    inference_steps: int = Field(25, ge=10, le=60)

# ---------------- UTILS ----------------

def download_file(url: str, save_path: Path):
    """Helper to download files from URLs to local disk"""
    try:
        # Added timeout to prevent hanging forever
        with requests.get(url, stream=True, timeout=120) as r:
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
        "--video_path", str(local_video_path),
        "--audio_path", str(local_audio_path),
        "--video_out_path", str(output_path)
    ]

    print(f"🚀 Running command: {' '.join(cmd)}")

    # We remove PIPE so you can see the progress bar in the Lightning Terminal
    process = subprocess.run(
        cmd,
        stdout=None, 
        stderr=None,
        text=True
    )
    
    if process.returncode != 0:
        raise RuntimeError(f"Inference Script Failed with code {process.returncode}")

# ---------------- BACKGROUND WORKER ----------------

def process_job(req: LipSyncRequest):
    """This runs in the background AFTER the API responds."""
    job_id = req.main_job_id
    print(f"[{job_id}] Processing Started...")
    
    # Define paths
    local_video_path = TEMP_DIR / f"{job_id}_video.mp4"
    local_audio_path = TEMP_DIR / f"{job_id}_audio.wav"
    final_output_path = OUTPUT_DIR / f"{job_id}.mp4"

    try:
        # 1. Download Inputs
        print(f"[{job_id}] Downloading files...")
        download_file(str(req.video_url), local_video_path)
        download_file(str(req.audio_url), local_audio_path)

        # 2. Run Inference
        print(f"[{job_id}] Running Inference...")
        run_lipsync(
            local_video_path,
            local_audio_path,
            final_output_path,
            req.inference_steps,
            req.guidance_scale
        )

        # 3. Construct Public URL
        # IMPORTANT: Ensure WORKER_PUBLIC_URL is set correctly at the top
        generated_url = f"{WORKER_PUBLIC_URL}/outputs/{job_id}.mp4"
        print(f"[{job_id}] Success! Video at: {generated_url}")

        # 4. Send Success Webhook
        print(f"[{job_id}] Sending Webhook to {req.webhook_url}...")
        requests.post(str(req.webhook_url), json={
            "job_id": job_id,
            "status": "completed",
            "generated_video_url": generated_url
        })

    except Exception as e:
        print(f"[{job_id}] FAILED: {str(e)}")
        
        # 5. Send Failure Webhook
        try:
            requests.post(str(req.webhook_url), json={
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            })
        except:
            print(f"[{job_id}] Could not send failure webhook.")

    finally:
        # 6. Cleanup Temp Files
        if local_video_path.exists(): os.remove(local_video_path)
        if local_audio_path.exists(): os.remove(local_audio_path)
        print(f"[{job_id}] Cleanup Complete.")

# ---------------- ROUTES ----------------

@app.get('/')
def root():
    return {"msg": "LatentSync Worker is Running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(req: LipSyncRequest, background_tasks: BackgroundTasks):
    """
    Receives request, queues it, and returns IMMEDIATELY.
    """
    # Add the processing function to the background queue
    background_tasks.add_task(process_job, req)
    
    return {
        "status": "queued",
        "message": "Job accepted. Processing in background.",
        "job_id": req.main_job_id
    }