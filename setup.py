#!/usr/bin/env python3
# Setup script for real-time diarization system

import os
import platform
import subprocess
import sys

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
def check_system_dependencies():
    """Check and install system dependencies if possible"""
    print("Checking system dependencies...")
    system = platform.system().lower()
    
    # Check for ffmpeg
    try:
        subprocess.check_call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ ffmpeg found")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ ffmpeg not found")
        if system == "darwin":  # macOS
            print("Installing ffmpeg via Homebrew...")
            try:
                subprocess.check_call(["brew", "install", "ffmpeg"])
            except (subprocess.SubprocessError, FileNotFoundError):
                print("Could not install ffmpeg automatically. Please install manually:")
                print("  brew install ffmpeg")
        elif system == "linux":
            print("Please install ffmpeg using your distribution's package manager:")
            print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("  Fedora: sudo dnf install ffmpeg")
        elif system == "windows":
            print("Please install ffmpeg manually:")
            print("  Download from https://ffmpeg.org/download.html")
            print("  Or using Chocolatey: choco install ffmpeg")
    
    # Check for PortAudio (required by PyAudio)
    if system == "darwin" or system == "linux":
        try:
            # This is a simplistic check - we're just seeing if the file exists
            found = False
            for path in ["/usr/local/lib/libportaudio.dylib", "/usr/lib/libportaudio.so", "/usr/local/lib/libportaudio.so"]:
                if os.path.exists(path):
                    found = True
                    break
            
            if found:
                print("✅ PortAudio found")
            else:
                print("❌ PortAudio not found")
                if system == "darwin":
                    print("Installing PortAudio via Homebrew...")
                    try:
                        subprocess.check_call(["brew", "install", "portaudio"])
                    except (subprocess.SubprocessError, FileNotFoundError):
                        print("Could not install PortAudio automatically. Please install manually:")
                        print("  brew install portaudio")
                elif system == "linux":
                    print("Please install PortAudio using your distribution's package manager:")
                    print("  Ubuntu/Debian: sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev")
                    print("  Fedora: sudo dnf install portaudio portaudio-devel")
        except Exception as e:
            print(f"Error checking for PortAudio: {e}")
            print("Please ensure PortAudio is installed manually")

def install_openai_whisper():
    """Explicitly install OpenAI Whisper from GitHub source"""
    print("Installing OpenAI Whisper from GitHub...")
    try:
        # First remove any existing whisper packages
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "whisper", "openai-whisper"])
        # Install directly from GitHub
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"])
        print("✅ OpenAI Whisper installed successfully")
    except subprocess.SubprocessError as e:
        print(f"❌ Error installing OpenAI Whisper: {e}")
        print("Please try installing manually:")
        print("  pip install git+https://github.com/openai/whisper.git")

def main():
    """Main setup function"""
    print("Setting up real-time diarization system...")
    
    # Create a virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        
        # Activate the virtual environment
        if platform.system().lower() == "windows":
            activate_script = os.path.join("venv", "Scripts", "activate")
        else:
            activate_script = os.path.join("venv", "bin", "activate")
        
        print(f"Virtual environment created. To activate, run:")
        print(f"  source {activate_script}")
        print("Then run this setup script again.")
        return
    
    # Check system dependencies
    check_system_dependencies()
    
    # Install Python dependencies - but skip Whisper for now
    try:
        print("Installing basic Python dependencies...")
        packages = [
            "numpy>=1.20.0", 
            "torch>=1.12.0", 
            "torchaudio>=0.12.0",
            "sounddevice>=0.4.5", 
            "speechbrain>=0.5.13",
            "matplotlib>=3.5.0", 
            "scipy>=1.7.0", 
            "tqdm>=4.62.0",
            "librosa>=0.9.0", 
            "pyaudio>=0.2.11", 
            "ffmpeg-python>=0.2.0"
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    except subprocess.SubprocessError as e:
        print(f"Warning: Error installing some packages: {e}")
        print("Continuing with setup...")
    
    # Install Whisper separately
    install_openai_whisper()
    
    print("\nSetup complete! You can now run the diarization system:")
    print("  python main.py")

if __name__ == "__main__":
    main() 