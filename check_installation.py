#!/usr/bin/env python3
# Verify installation and diagnose issues

import sys
import platform
import subprocess
import os
from importlib.util import find_spec

def check_package(package_name, expected_version=None, alternative_names=None):
    """Check if a package is installed and can be imported"""
    if alternative_names is None:
        alternative_names = []
    
    names_to_try = [package_name] + alternative_names
    
    for name in names_to_try:
        spec = find_spec(name)
        if spec is not None:
            try:
                module = __import__(name)
                version = getattr(module, "__version__", "unknown")
                print(f"✅ {package_name} found (version: {version})")
                
                if expected_version and version != "unknown":
                    if version < expected_version:
                        print(f"⚠️  {package_name} version {version} is older than expected {expected_version}")
                
                return True
            except ImportError:
                continue
    
    print(f"❌ {package_name} not found")
    return False

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              text=True)
        version_line = result.stdout.split('\n')[0]
        print(f"✅ ffmpeg found: {version_line}")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("❌ ffmpeg not found - audio processing may not work correctly")
        return False

def check_portaudio():
    """Check for portaudio library"""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        lib_paths = ["/usr/local/lib/libportaudio.dylib", 
                   "/opt/homebrew/lib/libportaudio.dylib"]
    elif system == "linux":
        lib_paths = ["/usr/lib/libportaudio.so", 
                   "/usr/local/lib/libportaudio.so"]
    else:  # Windows
        lib_paths = []  # Windows handles things differently
    
    for path in lib_paths:
        if os.path.exists(path):
            print(f"✅ PortAudio found at {path}")
            return True
    
    # Try importing pyaudio as another check
    try:
        import pyaudio
        print(f"✅ PyAudio found (implying PortAudio is working)")
        return True
    except ImportError:
        pass
    
    print("❌ PortAudio not found - microphone access may not work")
    return False

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Device: {device_name}")
            return True
        else:
            print("❌ CUDA not available - using CPU only")
            return False
    except ImportError:
        print("❌ PyTorch not found - cannot check CUDA")
        return False

def check_openai_whisper():
    """Specifically check for OpenAI's Whisper package"""
    try:
        import whisper
        # Try to access a function specific to OpenAI's implementation
        if hasattr(whisper, 'load_model'):
            print(f"✅ OpenAI Whisper found")
            try:
                tiny_model = whisper.load_model("tiny")
                print(f"✅ Whisper model loading works")
            except Exception as e:
                print(f"⚠️  Whisper model loading failed: {e}")
            return True
        else:
            print("❌ Found a 'whisper' package but it doesn't appear to be OpenAI Whisper")
            return False
    except ImportError:
        print("❌ OpenAI Whisper not found")
        return False

def main():
    """Main function to check all dependencies"""
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("\nChecking required packages:")
    
    packages_to_check = [
        ("numpy", "1.20.0"),
        ("torch", "1.12.0"),
        ("torchaudio", "0.12.0"),
        ("speechbrain", "0.5.13"),
        ("sounddevice", "0.4.5"),
        ("matplotlib", "3.5.0"),
        ("scipy", "1.7.0"),
    ]
    
    missing = []
    for package, version in packages_to_check:
        if not check_package(package, version):
            missing.append(package)
    
    print("\nChecking system dependencies:")
    check_ffmpeg()
    check_portaudio()
    
    print("\nChecking CUDA availability:")
    check_cuda()
    
    print("\nChecking OpenAI Whisper:")
    check_openai_whisper()
    
    if missing:
        print("\n⚠️  Some packages are missing. Run setup.py to install them:")
        print("  python setup.py")
    else:
        print("\n✅ All core packages found!")
    
    print("\nTo run the diarization system:")
    print("  python main.py")

if __name__ == "__main__":
    main() 