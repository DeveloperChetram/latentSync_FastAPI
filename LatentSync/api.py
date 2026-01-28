import os
import subprocess
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid

app = FastAPI()

# Mount the 'outputs' folder so we can serve the generated videos via URL
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Define the data we expect from your server
class GenerationRequest(BaseModel):
    video_url: str
    audio_url: str
    guidance_scale: float = 1.5  # Default tuned for quality
    inference_steps: int = 25    # Default tuned for L4 speed/quality

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download: {url}")
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@app.get('/')
def root_route():
    return {"msg":"server is running"}

@app.post("/generate")
async def generate_lipsync(request: GenerationRequest):
    # 1. Setup unique filenames to prevent collisions
    job_id = str(uuid.uuid4())
    temp_dir = "temp_api"
    os.makedirs(temp_dir, exist_ok=True)
    
    input_video_path = f"{temp_dir}/{job_id}_input.mp4"
    input_audio_path = f"{temp_dir}/{job_id}_input.mp3"
    output_video_name = f"{job_id}_output.mp4"
    output_video_path = f"outputs/{output_video_name}"

    try:
        # 2. Download files
        print(f"Downloading video: {request.video_url}")
        download_file(request.video_url, input_video_path)
        
        print(f"Downloading audio: {request.audio_url}")
        download_file(request.audio_url, input_audio_path)

        # 3. Construct the Inference Command
        # We use the settings optimized for your L4 (High Res 512 + 1.5 Guidance)
        command = [
            "python", "-m", "scripts.inference",
            "--unet_config_path", "configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--inference_steps", str(request.inference_steps),
            "--guidance_scale", str(request.guidance_scale),
            "--video_path", input_video_path,
            "--audio_path", input_audio_path,
            "--video_out_path", output_video_path
        ]

        # 4. Run Inference
        print("Running LatentSync...")
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")

        # 5. Return the Download Link
        # NOTE: You will replace 'YOUR_LIGHTNING_URL' in the next step
        download_link = f"/outputs/{output_video_name}"
        return {
            "status": "success", 
            "video_link": download_link,
            "job_id": job_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup input files (optional)
        if os.path.exists(input_video_path):
            os.remove(input_video_path)
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)