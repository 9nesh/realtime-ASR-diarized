#!/usr/bin/env python3
# Real-Time Speaker Transcription and Diarization System
# This script implements a real-time, incremental approach to speaker diarization

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import sys

# Ensure we import the correct Whisper package
if 'whisper' in sys.modules:
    del sys.modules['whisper']
import whisper

import queue
import threading
import time
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

# Import our modules for embedding and clustering
from ecapa_embedding import ECAPAEmbedder
from incremental_clustering import IncrementalClustering, Speaker

# Import SpeechBrain for ECAPA-TDNN
try:
    from speechbrain.inference import EncoderClassifier
except ImportError:
    print("SpeechBrain not found. Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "speechbrain"])
    from speechbrain.inference import EncoderClassifier

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RealTimeDiarizer")

# Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.25)  # Smaller chunks for more frequent processing
SILENCE_THRESHOLD = 0.003  # Lower threshold to detect more speech
MIN_SPEECH_DURATION = 0.5  # Minimum speech duration in seconds
MAX_AUDIO_BUFFER = int(SAMPLE_RATE * 4)  # Maximum 4 seconds in buffer
MAX_SILENCE_KEEP = int(SAMPLE_RATE * 0.5)  # Keep up to 0.5 seconds of silence in buffer

@dataclass
class TranscriptSegment:
    """Class for storing a single transcript segment"""
    speaker_id: str
    text: str
    start_time: float
    end_time: float = 0.0
    is_final: bool = False

class RealTimeDiarizer:
    def __init__(self, 
                 whisper_model="base", 
                 device="cpu",
                 similarity_threshold=0.65):
        logger.info("Initializing Real-Time Diarizer...")
        
        # Initialize components
        logger.info(f"Loading ASR model (Whisper {whisper_model})...")
        self.asr_model = whisper.load_model(whisper_model, device=device)
        self.device = device
        
        logger.info("Loading speaker embedding model (ECAPA-TDNN)...")
        # Use our dedicated embedder class
        self.embedder = ECAPAEmbedder()
        
        # Initialize the incremental clustering with a lower threshold
        self.clustering = IncrementalClustering(
            similarity_threshold=similarity_threshold,
            debug=True
        )
        
        # Audio processing
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_chunks = queue.Queue()
        self.transcript_buffer = []
        
        # For performance tracking
        self.processed_chunks = 0
        self.detected_speech_chunks = 0
        
        # Speaker tracking
        self.last_speaker_time = 0
        self.consecutive_short_segments = 0
        
        # For silence tracking
        self.silence_counter = 0
        self.speech_energy_history = deque(maxlen=20)  # Track recent energy levels
        self.last_process_time = 0
        self.dynamic_silence_threshold = SILENCE_THRESHOLD  # Will be adjusted dynamically
        
        # For duplicate detection
        self.last_transcription = ""
        self.transcription_history = deque(maxlen=5)  # Store recent transcriptions to avoid duplicates
        
        # For visualization
        self.should_stop = False
        logger.info("Initialization complete!")
    
    def start_audio_capture(self):
        """Start capturing audio from the microphone"""
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio capture error: {status}")
            
            # Add the new audio data to the buffer
            audio_data = indata.copy().reshape(-1).astype(np.float32)
            self.audio_chunks.put(audio_data)
        
        try:
            self.stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE
            )
            self.stream.start()
            logger.info("Audio capture started")
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
            logger.info("Make sure your microphone is connected and working.")
            raise
    
    def _update_dynamic_threshold(self, energy):
        """Update the dynamic silence threshold based on recent audio energy"""
        self.speech_energy_history.append(energy)
        
        if len(self.speech_energy_history) >= 10:
            # Set threshold to be a fraction of the 10th percentile energy
            sorted_energies = sorted(self.speech_energy_history)
            percentile_10 = sorted_energies[len(sorted_energies) // 10]
            
            # Make threshold adaptive but with limits
            self.dynamic_silence_threshold = max(SILENCE_THRESHOLD, min(percentile_10 * 0.5, 0.01))
            
            if self.processed_chunks % 100 == 0:
                logger.debug(f"Updated dynamic silence threshold to {self.dynamic_silence_threshold:.6f}")
    
    def _is_duplicate_transcription(self, text):
        """Check if a transcription is a duplicate of recent ones"""
        if not text:
            return True
            
        # Check exact match with recent transcriptions
        if text in self.transcription_history:
            return True
            
        # Check if text is a subset of recent transcriptions
        for recent in self.transcription_history:
            if text in recent:
                return True
                
        return False
    
    def process_audio(self):
        """Process audio chunks from the queue"""
        self.silence_counter = 0
        self.last_process_time = time.time()
        processing_delay = 0.5  # Start with 0.5 second processing delay
        
        while not self.should_stop:
            try:
                # Get audio chunk from the queue with timeout
                try:
                    chunk = self.audio_chunks.get(timeout=0.1)
                    self.processed_chunks += 1
                except queue.Empty:
                    # If no new audio for a while, process any accumulated audio
                    if len(self.audio_buffer) > SAMPLE_RATE * 1.0 and time.time() - self.last_process_time > 1.0:
                        logger.debug("Processing accumulated audio after queue timeout")
                        self.process_chunk(self.audio_buffer)
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.last_process_time = time.time()
                    continue
                
                # Calculate energy for this chunk
                energy = np.sqrt(np.mean(chunk**2))
                self._update_dynamic_threshold(energy)
                
                # Voice Activity Detection with dynamic threshold
                if energy < self.dynamic_silence_threshold:
                    self.silence_counter += 1
                    
                    # After consecutive silence, process any speech before clearing
                    if self.silence_counter >= 4:  # About 1 second of silence with smaller chunks
                        if len(self.audio_buffer) > SAMPLE_RATE * 1.0:
                            logger.debug(f"Processing buffer before silence (length: {len(self.audio_buffer)/SAMPLE_RATE:.2f}s)")
                            self.process_chunk(self.audio_buffer)
                            # Keep a small amount of silence for context - using np.zeros instead of slicing chunk
                            self.audio_buffer = np.zeros(MAX_SILENCE_KEEP, dtype=np.float32)
                            self.last_process_time = time.time()
                    
                    if self.processed_chunks % 40 == 0:  # Log less frequently 
                        logger.debug(f"Silence detected (energy: {energy:.6f}, threshold: {self.dynamic_silence_threshold:.6f})")
                        
                    # Still add some silence to buffer for context
                    if len(self.audio_buffer) < MAX_AUDIO_BUFFER:
                        self.audio_buffer = np.append(self.audio_buffer, chunk)
                else:
                    # Reset silence counter on speech
                    self.silence_counter = 0
                    
                    # Add chunk to audio buffer
                    self.audio_buffer = np.append(self.audio_buffer, chunk)
                    
                    # Dynamically adjust processing delay based on audio length
                    buffer_duration = len(self.audio_buffer) / SAMPLE_RATE
                    if buffer_duration >= 3.0:
                        processing_delay = 0.5  # Process sooner for long buffers
                    else:
                        processing_delay = 1.25  # Longer delay for short buffers
                    
                    # Process the audio if enough time has passed since last processing
                    if time.time() - self.last_process_time > processing_delay and buffer_duration >= 1.0:
                        self.detected_speech_chunks += 1
                        logger.debug(f"Speech detected (energy: {energy:.6f}), processing buffer of {buffer_duration:.2f}s")
                        self.process_chunk(self.audio_buffer)
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.last_process_time = time.time()
                
                # Prevent buffer from getting too large
                if len(self.audio_buffer) > MAX_AUDIO_BUFFER:
                    logger.debug(f"Audio buffer too large ({len(self.audio_buffer)/SAMPLE_RATE:.2f}s), processing")
                    self.process_chunk(self.audio_buffer)
                    self.audio_buffer = np.array([], dtype=np.float32)
                    self.last_process_time = time.time()
                
            except queue.Empty:
                # Handle in the inner try block now
                pass
            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
    
    def process_chunk(self, audio_data: np.ndarray):
        """Process a single chunk of audio"""
        if len(audio_data) < MIN_SPEECH_DURATION * SAMPLE_RATE:
            logger.debug(f"Audio too short ({len(audio_data)/SAMPLE_RATE:.2f}s), skipping")
            return
        
        # Transcribe the audio
        start_time = time.time()
        transcript = self.transcribe_audio(audio_data)
        transcribe_time = time.time() - start_time
        
        if not transcript or not transcript.strip():
            logger.debug(f"No transcript detected (length: {len(audio_data)/SAMPLE_RATE:.2f}s)")
            # No transcript, keep collecting audio
            if len(audio_data) > 5 * SAMPLE_RATE:  # If audio is too large (5 seconds)
                logger.info("Clearing large audio buffer with no transcript")
            return
        
        # Check for duplicate transcription
        if self._is_duplicate_transcription(transcript):
            logger.debug(f"Skipping duplicate transcription: {transcript}")
            return
            
        # Add to transcription history
        self.transcription_history.append(transcript)
        self.last_transcription = transcript
        
        logger.debug(f"Transcribed in {transcribe_time:.2f}s: {transcript}")
        
        # Extract speaker embedding - this is a critical part
        start_time = time.time()
        embedding = self.extract_speaker_embedding(audio_data)
        embedding_time = time.time() - start_time
        
        if embedding is None:
            logger.warning("Failed to extract speaker embedding")
            return
        
        logger.debug(f"Extracted embedding in {embedding_time:.2f}s")
        
        # Calculate segment duration
        segment_duration = len(audio_data) / SAMPLE_RATE
        
        # Force a new speaker if it's been a while since the last utterance
        # This helps prevent misattributing speech after long pauses
        force_new_speaker = False
        current_time = time.time()
        if current_time - self.last_speaker_time > 5.0:  # 5 second gap indicates potential speaker change
            force_new_speaker = True
            logger.debug("Long pause detected, forcing potential speaker change")
        self.last_speaker_time = current_time
        
        # Enhanced speaker detection for short segments
        if segment_duration < 1.0:
            self.consecutive_short_segments += 1
        else:
            self.consecutive_short_segments = 0
            
        # Determine the speaker using our improved clustering
        start_time = time.time()
        speaker_id, confidence = self.clustering.predict_speaker(embedding, segment_duration)
        clustering_time = time.time() - start_time
        
        logger.debug(f"Speaker clustering took {clustering_time:.2f}s, assigned to {speaker_id} with confidence {confidence:.3f}")
        
        # Add to transcript buffer
        segment = TranscriptSegment(
            speaker_id=speaker_id,
            text=transcript,
            start_time=current_time - segment_duration,
            end_time=current_time,
            is_final=True
        )
        self.transcript_buffer.append(segment)
        
        # Print the transcript with speaker information
        self.print_transcript(segment)
    
    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper"""
        try:
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # Pad if too short
            if len(audio) < 0.5 * SAMPLE_RATE:
                padding = np.zeros(int(0.5 * SAMPLE_RATE) - len(audio), dtype=np.float32)
                audio = np.concatenate([audio, padding])
            
            # Whisper transcription
            result = self.asr_model.transcribe(audio, 
                                              language="en", 
                                              fp16=(self.device == "cuda"))
            
            text = result["text"].strip()
            return text
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""
    
    def extract_speaker_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract speaker embedding using ECAPA-TDNN"""
        try:
            # Use our dedicated embedder
            embedding = self.embedder.extract_embedding(audio)
            return embedding
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return None
    
    def print_transcript(self, segment: TranscriptSegment):
        """Print transcript segment with speaker ID"""
        # Get speaker info from our clustering module
        speaker_names = self.clustering.get_speaker_names()
        speaker_name = speaker_names.get(segment.speaker_id, f"Speaker {segment.speaker_id}")
        
        # Print with timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime(segment.start_time))
        print(f"[{timestamp}] {speaker_name}: {segment.text}")
        
        # Log some stats occasionally
        if self.processed_chunks % 100 == 0:
            logger.info(f"Stats: {self.detected_speech_chunks}/{self.processed_chunks} chunks contained speech")
            logger.info(f"Currently tracking {self.clustering.get_num_speakers()} speakers")
            logger.info(f"Current silence threshold: {self.dynamic_silence_threshold:.6f}")
    
    def stop(self):
        """Stop the diarizer"""
        self.should_stop = True
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        logger.info("Diarization stopped")

    def run(self):
        """Run the real-time diarization"""
        try:
            self.start_audio_capture()
            
            # Start audio processing in a separate thread
            processing_thread = threading.Thread(target=self.process_audio)
            processing_thread.daemon = True
            processing_thread.start()
            
            logger.info("Real-time diarization is running. Press Ctrl+C to stop.")
            
            try:
                # Keep the main thread alive
                while not self.should_stop:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Stopping diarization...")
            finally:
                self.stop()
                processing_thread.join(timeout=2.0)
                
            # Print final stats
            logger.info("\n--- Final Statistics ---")
            logger.info(f"Total audio chunks processed: {self.processed_chunks}")
            logger.info(f"Chunks with speech detected: {self.detected_speech_chunks}")
            logger.info(f"Total speakers identified: {self.clustering.get_num_speakers()}")
            
            speaker_stats = self.clustering.get_speaker_stats()
            for speaker_id, stats in speaker_stats.items():
                logger.info(f"Speaker {stats['name']}: {stats['speech_segments']} segments, {stats['total_duration']:.1f}s total duration")
                
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            raise

if __name__ == "__main__":
    diarizer = RealTimeDiarizer(
        whisper_model="base",
        similarity_threshold=0.60  # Lower threshold to detect different speakers more easily
    )
    diarizer.run() 