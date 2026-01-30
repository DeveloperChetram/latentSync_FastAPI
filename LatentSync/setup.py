import os
import subprocess
import sys
import shutil

def run_command(command, description):
    print(f"🚀 {description}...")
    try:
        subprocess.check_call(command, shell=True)
        print(f"✅ {description} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        sys.exit(1)

def install_python_packages():
    """Installs pip dependencies."""
    if os.path.exists("requirements.txt"):
        run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages")
    else:
        print("⚠️ requirements.txt not found! Skipping package installation.")

def install_ffmpeg_static():
    """Downloads and installs FFmpeg static build with libx264."""
    print("🎬 Checking FFmpeg Static Installation...")
    
    # Configuration
    download_url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    tar_filename = "ffmpeg-release-amd64-static.tar.xz"
    binary_name = "ffmpeg_static"
    
    # 1. Check if already installed
    if os.path.exists(binary_name):
        print(f"✅ {binary_name} found. Verifying permissions...")
        os.chmod(binary_name, 0o755) # Ensure executable
        return

    # 2. Download
    print("⬇️  Downloading FFmpeg static build (Includes libx264)...")
    run_command(f"wget {download_url}", "Downloading FFmpeg archive")

    # 3. Extract
    run_command(f"tar xvf {tar_filename}", "Extracting FFmpeg")

    # 4. Move Binary
    # The tar extracts to a folder like 'ffmpeg-7.0.2-amd64-static'
    # We find it dynamically
    extracted_dirs = [d for d in os.listdir(".") if d.startswith("ffmpeg-") and d.endswith("-static") and os.path.isdir(d)]
    
    if not extracted_dirs:
        print("❌ Error: Could not find extracted FFmpeg folder.")
        sys.exit(1)
        
    source_binary = os.path.join(extracted_dirs[0], "ffmpeg")
    
    if os.path.exists(binary_name):
        os.remove(binary_name)
        
    print(f"🚚 Moving binary to ./{binary_name}...")
    shutil.move(source_binary, binary_name)

    # 5. Make Executable
    print("🔑 Setting executable permissions...")
    os.chmod(binary_name, 0o755)

    # 6. Cleanup
    print("🧹 Cleaning up temporary files...")
    if os.path.exists(tar_filename):
        os.remove(tar_filename)
    if os.path.exists(extracted_dirs[0]):
        shutil.rmtree(extracted_dirs[0])

    print("✅ FFmpeg Static installation complete!")

def download_models():
    """Downloads the required AI models."""
    # Define model paths
    checkpoints_dir = "checkpoints"
    unet_path = os.path.join(checkpoints_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(checkpoints_dir, "whisper", "tiny.pt")

    os.makedirs(checkpoints_dir, exist_ok=True)

    # 1. Download LatentSync UNet (v1.6 High Res)
    if not os.path.exists(unet_path):
        cmd = "huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints --local-dir-use-symlinks False"
        run_command(cmd, "Downloading LatentSync v1.6 Model")
    else:
        print("✅ LatentSync Model already exists.")

    # 2. Download Whisper Model
    if not os.path.exists(whisper_path):
        cmd = "huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints --local-dir-use-symlinks False"
        run_command(cmd, "Downloading Whisper Model")
    else:
        print("✅ Whisper Model already exists.")

def main():
    print("=== 🛠️ LatentSync One-Click Setup 🛠️ ===\n")
    
    # Step 1: Install Python Libs
    install_python_packages()

    # Step 2: Install Special FFmpeg (The one with libx264)
    install_ffmpeg_static()

    # Step 3: Download AI Models
    download_models()

    print("\n🎉 Setup Complete! You are ready to generate videos.")

if __name__ == "__main__":
    main()