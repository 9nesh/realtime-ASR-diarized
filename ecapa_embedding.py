#!/usr/bin/env python3
# ECAPA-TDNN Speaker Embedding Extractor
# Optimized for real-time use

import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import warnings
from typing import Optional, Union, List
from dataclasses import dataclass

# Import ECAPA-TDNN model
try:
    from speechbrain.inference import EncoderClassifier
    import speechbrain as sb
except ImportError:
    print("SpeechBrain not found. Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "speechbrain"])
    from speechbrain.inference import EncoderClassifier
    import speechbrain as sb

# Setup logging
import logging
logger = logging.getLogger("ECAPAEmbedder")

@dataclass
class SpeakerEmbedding:
    """Container for speaker embedding data and metadata"""
    embedding: np.ndarray
    confidence: float = 1.0
    duration: float = 0.0
    timestamp: float = 0.0

class ECAPAEmbedder:
    """ECAPA-TDNN speaker embedding extractor optimized for real-time use"""
    
    def __init__(self, 
                 model_path: str = "speechbrain/spkrec-ecapa-voxceleb",
                 device: str = "cpu",
                 cache_dir: str = "./pretrained_models/ecapa-tdnn"):
        """Initialize the ECAPA-TDNN model
        
        Args:
            model_path: Path to the pretrained model (default is from HuggingFace)
            device: Device to run the model on ('cpu' or 'cuda')
            cache_dir: Directory to cache the model
        """
        logger.info(f"Loading ECAPA-TDNN model from {model_path}")
        
        # Filter torchaudio warnings - common with SpeechBrain
        warnings.filterwarnings("ignore", message=".*torchaudio backend.*")
        
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load the model
            self.encoder = EncoderClassifier.from_hparams(
                source=model_path,
                savedir=cache_dir,
                run_opts={"device": device}
            )
            
            self.device = device
            logger.info("ECAPA-TDNN model loaded successfully")
            
            # Keep track of statistics
            self.num_embeddings_extracted = 0
            self.avg_extraction_time = 0
            
        except Exception as e:
            logger.error(f"Error loading ECAPA-TDNN model: {e}")
            raise
    
    def preprocess_audio(self, audio: Union[np.ndarray, torch.Tensor], 
                           sample_rate: int = 16000) -> torch.Tensor:
        """Preprocess audio for the model
        
        Args:
            audio: Audio waveform (either numpy array or torch tensor)
            sample_rate: Sample rate of the audio (default: 16000)
            
        Returns:
            Preprocessed audio tensor
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(audio, np.ndarray):
            # Normalize audio if needed
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(np.abs(audio.max()), np.abs(audio.min()))
            
            # Convert to float32 tensor
            waveform = torch.tensor(audio, dtype=torch.float32)
        else:
            waveform = audio
            
        # Add batch dimension if needed
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Ensure correct shape: [batch, time]
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.transpose(0, 1)
            
        # Move to the correct device
        waveform = waveform.to(self.device)
        
        return waveform
    
    def extract_embedding(self, 
                          audio: Union[np.ndarray, torch.Tensor], 
                          sample_rate: int = 16000,
                          normalize: bool = True) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio
        
        Args:
            audio: Audio signal (can be numpy array or torch tensor)
            sample_rate: Sample rate of the audio
            normalize: Whether to normalize the embedding
            
        Returns:
            Speaker embedding as numpy array, or None if extraction fails
        """
        try:
            import time
            start_time = time.time()
            
            # Convert to the format expected by SpeechBrain
            if isinstance(audio, np.ndarray):
                # Normalize audio if it's not already
                if np.max(np.abs(audio)) > 1.0:
                    audio = audio / (np.max(np.abs(audio)) + 1e-10)
                
                # Convert to torch tensor
                waveform = torch.tensor(audio).float().unsqueeze(0)
            else:
                waveform = audio.unsqueeze(0) if audio.dim() == 1 else audio
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(waveform)
                
                # Normalize if requested (recommended)
                if normalize:
                    embedding = F.normalize(embedding, p=2, dim=1)
                
                # Convert to numpy
                embedding_np = embedding.squeeze().cpu().numpy()
            
            # Update stats
            self.num_embeddings_extracted += 1
            extraction_time = time.time() - start_time
            self.avg_extraction_time = (self.avg_extraction_time * (self.num_embeddings_extracted - 1) + 
                                       extraction_time) / self.num_embeddings_extracted
            
            if self.num_embeddings_extracted % 50 == 0:
                logger.debug(f"Extracted {self.num_embeddings_extracted} embeddings " +
                            f"(avg time: {self.avg_extraction_time*1000:.1f}ms)")
            
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return None
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Make sure embeddings are normalized
        if np.linalg.norm(emb1) > 1.01 or np.linalg.norm(emb2) > 1.01:
            emb1 = emb1 / np.linalg.norm(emb1)
            emb2 = emb2 / np.linalg.norm(emb2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2)
        
        # Ensure the result is between 0 and 1
        similarity = max(0, min(1, similarity))
        
        return similarity
    
    def compare_embeddings(self, embeddings1: List[Union[np.ndarray, SpeakerEmbedding]], 
                           embeddings2: List[Union[np.ndarray, SpeakerEmbedding]]) -> np.ndarray:
        """Compare two lists of embeddings and return similarity matrix
        
        Args:
            embeddings1: First list of embeddings
            embeddings2: Second list of embeddings
            
        Returns:
            Similarity matrix of shape [len(embeddings1), len(embeddings2)]
        """
        # Extract embedding arrays if needed
        embs1 = [e.embedding if isinstance(e, SpeakerEmbedding) else e for e in embeddings1]
        embs2 = [e.embedding if isinstance(e, SpeakerEmbedding) else e for e in embeddings2]
        
        # Convert to numpy arrays
        embs1 = np.vstack(embs1)
        embs2 = np.vstack(embs2)
        
        # Compute dot product
        sim_matrix = np.dot(embs1, embs2.T)
        
        # Normalize
        norms1 = np.linalg.norm(embs1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(embs2, axis=1, keepdims=True)
        sim_matrix = sim_matrix / np.dot(norms1, norms2.T)
        
        # Clip to [0, 1] range
        sim_matrix = np.clip(sim_matrix, 0, 1)
        
        return sim_matrix
    
    def batch_extract_embeddings(self, 
                                audio_segments: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extract embeddings for multiple audio segments in batch
        
        Args:
            audio_segments: List of audio segments
            
        Returns:
            List of embeddings (or None for failed extractions)
        """
        embeddings = []
        for audio in audio_segments:
            embedding = self.extract_embedding(audio)
            embeddings.append(embedding)
        return embeddings

# Example usage
if __name__ == "__main__":
    # Test the embedding extractor with a random audio signal
    print("Testing ECAPA-TDNN embedding extractor...")
    
    # Create a random audio signal (white noise)
    sample_rate = 16000
    duration = 3  # seconds
    audio = np.random.randn(sample_rate * duration).astype(np.float32)
    
    # Initialize the model
    embedder = ECAPAEmbedder()
    
    # Extract embedding
    print("Extracting embedding...")
    embedding = embedder.extract_embedding(audio)
    
    if embedding is not None:
        print(f"Embedding shape: {embedding.shape}")
        print(f"Extraction time: {time.time() - embedding.timestamp:.2f} seconds")
    else:
        print("Failed to extract embedding") 