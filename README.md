# Real-Time Speaker Transcription and Diarization

This project implements a real-time speech transcription and speaker diarization system that processes audio in real-time to:

1. Transcribe speech immediately using OpenAI's Whisper model
2. Dynamically identify and track speakers using ECAPA-TDNN embeddings
3. Generate a transcript with speaker separation

Unlike traditional diarization methods that require offline clustering of the full conversation, this system uses an incremental, online clustering algorithm to enable real-time speaker identification.

## Features

- **Real-time processing**: Audio is processed in small chunks with low latency
- **Streaming transcription**: Uses Whisper ASR model to transcribe speech as it happens
- **Dynamic speaker identification**: Incrementally clusters speaker embeddings
- **Voice activity detection**: Filters out silence and non-speech segments
- **Visualization**: (Optional) Real-time visualization of speaker activity
- **JSON output**: Structured output format for further processing

## System Architecture

The system consists of four main components:

1. **Audio Processing**: Capture and preprocess audio from the microphone
2. **Automatic Speech Recognition (ASR)**: Transcribe audio using OpenAI's Whisper
3. **Speaker Embedding**: Extract ECAPA-TDNN speaker embeddings
4. **Incremental Clustering**: Dynamically identify and track speakers

## Requirements

- Python 3.7+
- PyTorch 1.12+
- SpeechBrain (for ECAPA-TDNN embeddings)
- OpenAI Whisper (for ASR)
- SoundDevice (for audio capture)
- Additional requirements in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/real-time-diarization.git
   cd real-time-diarization
   ```

2. Run the setup script:
   ```
   python setup.py
   ```
   This script will:
   - Create a virtual environment (if needed)
   - Check for system dependencies (ffmpeg, PortAudio)
   - Install Python dependencies
   - Ensure the correct version of OpenAI Whisper is installed

3. Activate the virtual environment:
   ```
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Verify your installation:
   ```
   python check_installation.py
   ```
   This will check all dependencies and confirm that Whisper can load models correctly.

5. You're ready to run the system!

## Speaker Detection Tuning

To improve speaker detection in real-world scenarios, we've implemented several enhancements:

1. **Adaptive Similarity Threshold**: The default threshold of 0.65 works well for most scenarios, but you can adjust it:
   - Lower values (0.55-0.60) detect more speakers but may incorrectly separate the same speaker
   - Higher values (0.70-0.80) are more conservative and may group different speakers together
   - Use `--threshold 0.60` for better sensitivity to different speakers

2. **Silence-Based Segmentation**: The system uses silence detection to intelligently segment speech:
   - Short pauses maintain the same speaker
   - Longer pauses (>5 seconds) suggest a possible speaker change
   - Multiple consecutive short utterances are analyzed for potential speaker changes

3. **Forced Speaker Variety**: After several consecutive segments from the same speaker, the system will encourage speaker changes:
   - This helps prevent a single speaker from dominating the transcript
   - Particularly useful in multi-speaker environments with imbalanced speaking time

4. **Recency Weighting**: More recent speech from a speaker increases the likelihood of subsequent speech being attributed to them

If you have issues with speaker detection:

- Try using longer speech segments (>2 seconds) for more reliable speaker embedding extraction
- Keep consistent microphone distance for all speakers
- Minimize background noise and cross-talk
- Use the `--verbose` flag to see detailed logs about speaker detection

## Troubleshooting

### Common Issues

1. **Wrong Whisper Package**:
   ```
   AttributeError: module 'whisper' has no attribute 'load_model'
   ```
   This happens when the wrong "whisper" package is installed. Run:
   ```
   pip uninstall whisper
   pip install git+https://github.com/openai/whisper.git
   ```

2. **Whisper Installation Fails**:
   If you're having issues installing Whisper via pip, try installing it directly from GitHub:
   ```
   pip install git+https://github.com/openai/whisper.git
   ```
   
   If you continue to have issues, you may need to ensure you have the required build tools:
   - **Windows**: Install Visual C++ Build Tools
   - **Linux**: `sudo apt-get install build-essential`
   - **macOS**: `xcode-select --install`

3. **Missing Audio Backend**:
   ```
   SpeechBrain could not find any working torchaudio backend
   ```
   Ensure you have ffmpeg installed:
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
   - **Windows**: Download from ffmpeg.org or use `choco install ffmpeg`

4. **PyAudio Installation Issues**:
   If you encounter errors installing PyAudio, you may need to install PortAudio first:
   - **macOS**: `brew install portaudio`
   - **Ubuntu/Debian**: `sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev`
   - **Windows**: Check PyAudio documentation for Windows installation

5. **SpeechBrain Module Deprecation Warnings**:
   If you see warnings about `speechbrain.pretrained` being deprecated, update imports to use `speechbrain.inference` instead.

6. **No microphone detected**:
   If the system can't access your microphone, ensure:
   - Your microphone is properly connected and working with other applications
   - You've granted the necessary permissions to access the microphone
   - You've installed the required packages for audio capture (PyAudio and PortAudio)

7. **Speaker Detection Issues**:
   - Only detecting one speaker: Try using `--threshold 0.60` or even lower (0.55)
   - Incorrectly separating the same speaker: Try using `--threshold 0.70` or higher 
   - Speakers talking at different volumes: Try to maintain consistent microphone distance

## Usage

### Basic Usage

Run the main script to start real-time diarization:

```bash
python main.py
```

This will:
1. Initialize the models
2. Start audio capture from your default microphone
3. Begin real-time transcription and diarization
4. Save the transcript to `transcript.json`

Speak into your microphone, and the system will transcribe your speech and identify different speakers. Press Ctrl+C to stop.

### Command Line Options

```
python main.py --help
```

Available options:
- `--model`: Whisper model size to use (`tiny`, `base`, `small`, `medium`, `large`). Default: `base`
- `--device`: Device to run models on (`cpu` or `cuda`). Default: `cpu`
- `--threshold`: Speaker similarity threshold (0.0-1.0). Default: 0.65
- `--output`: Output file path. Default: `transcript.json`
- `--visualize`: Enable visualization of speaker activity (requires matplotlib)
- `--verbose`: Enable verbose logging for debugging

Examples:
```bash
# Use the tiny model for faster processing
python main.py --model tiny

# Use CUDA for GPU acceleration
python main.py --device cuda

# Lower the threshold for better speaker detection
python main.py --threshold 0.60

# Save output to a custom file
python main.py --output my_transcript.json

# Enable visualization and verbose logging
python main.py --visualize --verbose
```

## Output Format

The transcript is saved in JSON format with the following structure:

```json
{
  "metadata": {
    "start_time": 1626981234.56,
    "duration": 300.5,
    "speakers": 2,
    "date": "2023-06-15T14:30:45.123456"
  },
  "segments": [
    {
      "speaker_id": "S0",
      "speaker_name": "Speaker 1",
      "text": "Hello, how are you today?",
      "start_time": 1626981235.1,
      "end_time": 1626981237.8,
      "confidence": 0.95,
      "language": "en"
    },
    ...
  ],
  "speakers": {
    "S0": {
      "name": "Speaker 1",
      "color": "#FF4B4B",
      "total_duration": 120.5,
      "speech_segments": 25,
      "last_active_seconds_ago": 5.2,
      "confidence": 0.98,
      "num_embeddings": 18
    },
    ...
  }
}
```

## Performance Considerations

- **Model Size**: Smaller Whisper models (`tiny`, `base`) are faster but less accurate
- **CPU vs GPU**: GPU acceleration is highly recommended for larger models
- **Audio Quality**: Clear audio with minimal background noise yields better results
- **Number of Speakers**: Performance may degrade with more than 4-5 speakers
- **Speaker Similarity**: Adjust the similarity threshold as needed for your use case

## Advanced Usage

### Using as a Module

You can import the components in your own Python code:

```python
from real_time_asr import RealTimeASR
from ecapa_embedding import ECAPAEmbedder
from incremental_clustering import IncrementalClustering

# Initialize components
asr = RealTimeASR(model_name="base")
embedder = ECAPAEmbedder()
clustering = IncrementalClustering()

# Process audio
# ... (see example in main.py)
```

## Limitations

- **Speaker Confusion**: Similar-sounding speakers may be incorrectly clustered
- **Speaker Overlap**: The system cannot handle multiple people speaking simultaneously
- **First Few Seconds**: Initial speech may be buffered until a speaker is identified
- **Resource Usage**: Larger models consume significant CPU/GPU resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for ASR
- [SpeechBrain](https://github.com/speechbrain/speechbrain) for ECAPA-TDNN embeddings
- [ECAPA-TDNN paper](https://arxiv.org/abs/2005.07143) by Desplanques et al. 