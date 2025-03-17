#!/usr/bin/env python3
# Real-Time Automatic Speech Recognition (ASR)
# Using OpenAI's Whisper model for streaming transcription

import numpy as np
import torch
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealTimeASR")

# Try to import OpenAI Whisper
try:
    # Explicitly unload any existing whisper module to avoid conflicts
    import sys
    if 'whisper' in sys.modules:
        del sys.modules['whisper']
        
    # Now import the correct OpenAI whisper package
    import whisper
except ImportError:
    logger.error("OpenAI Whisper not found. Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/whisper.git"])
    import whisper

@dataclass
class TranscriptionResult:
    """Container for ASR results and metadata"""
    text: str
    start_time: float
    end_time: float
    language: str = "en"
    segments: List[Dict] = None
    words: List[Dict] = None
    confidence: float = 0.0
    is_final: bool = False

class RealTimeASR:
    """Real-time Automatic Speech Recognition using Whisper"""
    
    def __init__(self, model_name: str = "base", 
                 device: str = "cpu",
                 language: str = "en",
                 compute_word_timings: bool = False,
                 use_vad: bool = True,
                 min_speech_prob: float = 0.5,
                 use_fp16: bool = False):
        """Initialize the Real-Time ASR module
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on ('cpu' or 'cuda')
            language: Language code for ASR (default: English)
            compute_word_timings: Whether to compute word-level timings (slower)
            use_vad: Whether to use voice activity detection
            min_speech_prob: Minimum probability for speech detection
            use_fp16: Whether to use half-precision (only for CUDA)
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.compute_word_timings = compute_word_timings
        self.use_vad = use_vad
        self.min_speech_prob = min_speech_prob
        self.use_fp16 = use_fp16 and device == "cuda"
        
        logger.info(f"Initializing Real-Time ASR with model: {model_name}")
        start_time = time.time()
        
        # Load the Whisper model
        self.model = whisper.load_model(model_name, device=device)
        
        # Set options for transcription
        self.options = whisper.DecodingOptions(
            language=language,
            without_timestamps=not compute_word_timings,
            fp16=self.use_fp16
        )
        
        # For audio buffering
        self.sample_rate = 16000  # Whisper uses 16kHz
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # For streaming
        self.processing_thread = None
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.should_stop = False
        
        # Callback for transcription results
        self.transcription_callback = None
        
        # To prevent duplicated transcriptions
        self.last_transcription = ""
        self.transcription_history = []
        
        # Performance metrics
        self.transcript_count = 0
        self.total_processing_time = 0
        self.processing_times = []
        
        logger.info(f"ASR model loaded in {time.time() - start_time:.2f} seconds")
    
    def set_callback(self, callback: Callable[[TranscriptionResult], None]):
        """Set callback function for transcription results
        
        Args:
            callback: Function to call with transcription results
        """
        self.transcription_callback = callback
    
    def add_audio(self, audio: np.ndarray):
        """Add audio to the buffer for processing
        
        Args:
            audio: Audio waveform (float32, scaled to [-1.0, 1.0])
        """
        with self.buffer_lock:
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Add to buffer
            self.audio_buffer = np.append(self.audio_buffer, audio)
            
            # If not streaming, process directly
            if self.processing_thread is None:
                return
            
            # Add to queue for streaming
            self.audio_queue.put(audio)
    
    def reset_buffer(self):
        """Clear the audio buffer"""
        with self.buffer_lock:
            self.audio_buffer = np.array([], dtype=np.float32)
    
    def transcribe_buffer(self) -> Optional[TranscriptionResult]:
        """Transcribe the current audio buffer
        
        Returns:
            TranscriptionResult or None if no speech detected
        """
        with self.buffer_lock:
            if len(self.audio_buffer) < 0.5 * self.sample_rate:
                # Too short to transcribe
                return None
            
            # Make a copy to avoid race conditions
            audio = self.audio_buffer.copy()
        
        # Call the internal transcribe function
        return self._transcribe_audio(audio)
    
    def _transcribe_audio(self, audio: np.ndarray) -> Optional[TranscriptionResult]:
        """Internal function to transcribe audio
        
        Args:
            audio: Audio waveform
            
        Returns:
            TranscriptionResult or None if no speech detected
        """
        start_time = time.time()
        
        try:
            # Check for voice activity if VAD is enabled
            if self.use_vad:
                speech_prob = self._check_speech_activity(audio)
                if speech_prob < self.min_speech_prob:
                    logger.debug(f"No speech detected (prob: {speech_prob:.2f})")
                    return None
            
            # Pad audio if needed
            if len(audio) < self.sample_rate:
                audio = np.pad(audio, (0, self.sample_rate - len(audio)))
            
            # Get full transcript
            result = self.model.transcribe(
                audio, 
                **{k: getattr(self.options, k) for k in ['language', 'without_timestamps', 'fp16']},
                temperature=0.0,  # Use greedy decoding for faster results
                word_timestamps=self.compute_word_timings
            )
            
            # Process the result
            text = result["text"].strip()
            
            # Skip if too similar to previous transcription
            if self._is_duplicate_transcription(text):
                logger.debug(f"Skipping duplicate transcription: {text}")
                return None
            
            # Create result object
            transcription = TranscriptionResult(
                text=text,
                start_time=time.time() - len(audio) / self.sample_rate,
                end_time=time.time(),
                language=result.get("language", self.language),
                segments=result.get("segments", []),
                words=result.get("words", []) if self.compute_word_timings else None,
                confidence=result.get("segments", [{}])[0].get("confidence", 0.0) if result.get("segments") else 0.0,
                is_final=True
            )
            
            processing_time = time.time() - start_time
            self.transcript_count += 1
            self.total_processing_time += processing_time
            self.processing_times.append(processing_time)
            
            # Track the last transcription to avoid duplicates
            self.last_transcription = text
            self.transcription_history.append(text)
            if len(self.transcription_history) > 5:
                self.transcription_history.pop(0)
            
            logger.debug(f"Transcribed in {processing_time:.2f}s: {text}")
            return transcription
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return None
    
    def _check_speech_activity(self, audio: np.ndarray) -> float:
        """Simple voice activity detection based on energy
        
        Args:
            audio: Audio waveform
            
        Returns:
            Probability of speech (0-1)
        """
        # This is a very basic VAD for demo purposes
        # In production, use a proper VAD model or endpoint detection
        
        # Calculate short-time energy
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        step = int(0.010 * self.sample_rate)  # 10ms step
        
        if len(audio) < frame_length:
            return 0.0
        
        frames = []
        for i in range(0, len(audio) - frame_length, step):
            frame = audio[i:i+frame_length]
            frames.append(np.mean(frame**2))
        
        energies = np.array(frames)
        
        # Calculate adaptive threshold
        threshold = 0.3 * np.mean(energies) if len(energies) > 0 else 0
        
        # Determine speech probability
        speech_frames = np.sum(energies > threshold)
        speech_prob = speech_frames / len(energies) if len(energies) > 0 else 0
        
        return speech_prob
    
    def _is_duplicate_transcription(self, text: str) -> bool:
        """Check if a transcription is too similar to recent ones
        
        Args:
            text: Transcription text
            
        Returns:
            True if duplicate, False otherwise
        """
        if not text:
            return True
            
        # Check exact match with last transcription
        if text == self.last_transcription:
            return True
        
        # Check if text is subset of recent transcriptions
        for recent in self.transcription_history:
            if text in recent:
                return True
        
        return False
    
    def start_streaming(self):
        """Start the streaming transcription mode"""
        if self.processing_thread is not None:
            logger.warning("Streaming already started")
            return
        
        self.should_stop = False
        self.processing_thread = threading.Thread(target=self._streaming_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        logger.info("Started streaming transcription")
    
    def stop_streaming(self):
        """Stop the streaming transcription mode"""
        if self.processing_thread is None:
            logger.warning("Streaming not started")
            return
            
        self.should_stop = True
        self.processing_thread.join(timeout=2.0)
        self.processing_thread = None
        logger.info("Stopped streaming transcription")
    
    def _streaming_worker(self):
        """Worker thread for streaming transcription"""
        # Buffer for accumulated audio
        audio_buffer = np.array([], dtype=np.float32)
        last_process_time = time.time()
        
        while not self.should_stop:
            try:
                # Get audio chunk from queue with timeout
                chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                audio_buffer = np.append(audio_buffer, chunk)
                
                # Check if buffer is long enough and enough time has passed
                buffer_duration = len(audio_buffer) / self.sample_rate
                time_since_last = time.time() - last_process_time
                
                if buffer_duration >= 2.0 or (buffer_duration >= 0.5 and time_since_last >= 2.0):
                    # Transcribe the buffer
                    result = self._transcribe_audio(audio_buffer)
                    
                    if result and result.text:
                        # Put result in queue
                        self.result_queue.put(result)
                        
                        # Call callback if set
                        if self.transcription_callback:
                            self.transcription_callback(result)
                    
                    # Keep some overlap for context
                    overlap = int(0.5 * self.sample_rate)
                    if len(audio_buffer) > overlap:
                        audio_buffer = audio_buffer[-overlap:]
                    
                    last_process_time = time.time()
                
            except queue.Empty:
                # No new audio
                pass
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
    
    def get_last_result(self) -> Optional[TranscriptionResult]:
        """Get the last transcription result
        
        Returns:
            Last transcription result or None if none available
        """
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        avg_time = self.total_processing_time / self.transcript_count if self.transcript_count > 0 else 0
        return {
            "transcript_count": self.transcript_count,
            "avg_processing_time": avg_time,
            "total_processing_time": self.total_processing_time,
            "real_time_factor": avg_time / 2.0 if avg_time > 0 else 0  # Assuming 2-second segments
        }

# Example usage
if __name__ == "__main__":
    # Create a test ASR instance
    asr = RealTimeASR(model_name="base")
    
    # Create sample audio data (white noise)
    duration = 3  # seconds
    sample_rate = 16000
    audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.1
    
    # Add some sine waves to simulate speech
    for freq in [100, 200, 300]:
        t = np.arange(0, duration, 1/sample_rate)
        audio += 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    print("Testing real-time ASR...")
    
    # Process audio
    asr.add_audio(audio)
    result = asr.transcribe_buffer()
    
    if result:
        print(f"Transcription: {result.text}")
        print(f"Processing time: {asr.get_performance_stats()['avg_processing_time']:.2f} seconds")
    else:
        print("No transcription result") 