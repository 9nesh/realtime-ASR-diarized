#!/usr/bin/env python3
# Incremental Speaker Clustering
# Real-time speaker identification without requiring the full conversation

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import threading
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IncrementalClustering")

@dataclass
class Speaker:
    """Class representing a speaker with their embeddings and metadata"""
    id: str  # Unique identifier
    name: str = ""  # Display name (can be updated)
    color: str = ""  # Color for visualization
    
    # Store recent embeddings in a deque for incremental updates
    embeddings: deque = field(default_factory=lambda: deque(maxlen=20))
    
    # Speech statistics
    total_duration: float = 0.0
    speech_segments: int = 0
    last_active: float = 0.0
    
    # Confidence estimation
    confidence: float = 0.0
    
    def add_embedding(self, embedding: np.ndarray, duration: float = 0.0):
        """Add a new embedding for this speaker"""
        self.embeddings.append(embedding)
        self.total_duration += duration
        self.speech_segments += 1
        self.last_active = time.time()
        
        # Update confidence based on number of samples
        self.confidence = min(0.5 + (len(self.embeddings) / 10), 0.99)
    
    def get_average_embedding(self) -> np.ndarray:
        """Get the average embedding vector for this speaker"""
        if not self.embeddings:
            return None
            
        # Compute average embedding
        return np.mean(np.array(self.embeddings), axis=0)
    
    def get_recency_weight(self) -> float:
        """Get a recency weight based on how recently this speaker was active"""
        time_since_active = time.time() - self.last_active
        # Weight decays exponentially with time
        return np.exp(-time_since_active / 60.0)  # 1-minute half-life
        
    def merge_with(self, other_speaker: 'Speaker'):
        """Merge another speaker's data into this one"""
        # Add embeddings
        for emb in other_speaker.embeddings:
            if len(self.embeddings) < self.embeddings.maxlen:
                self.embeddings.append(emb)
                
        # Update statistics
        self.total_duration += other_speaker.total_duration
        self.speech_segments += other_speaker.speech_segments
        self.last_active = max(self.last_active, other_speaker.last_active)
        
        # Recalculate confidence
        self.confidence = min(0.5 + (len(self.embeddings) / 10), 0.99)


class IncrementalClustering:
    """Incremental speaker clustering algorithm for real-time diarization"""
    
    def __init__(self, similarity_threshold: float = 0.65,
                 max_speakers: int = 10,
                 min_embeddings_for_reliability: int = 3,
                 use_adaptive_threshold: bool = True,
                 debug: bool = True):
        """Initialize the incremental clustering algorithm
        
        Args:
            similarity_threshold: Cosine similarity threshold for speaker matching
            max_speakers: Maximum number of speakers to track
            min_embeddings_for_reliability: Minimum number of embeddings needed for reliable identification
            use_adaptive_threshold: Whether to adapt the threshold based on speaker confidence
            debug: Enable debug logging
        """
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.min_embeddings_for_reliability = min_embeddings_for_reliability
        self.use_adaptive_threshold = use_adaptive_threshold
        self.debug = debug
        
        # Speaker tracking
        self.speakers: Dict[str, Speaker] = {}
        self.next_speaker_id = 0
        
        # Default colors for visualization
        self.colors = ['#FF4B4B', '#4B7BFF', '#37C871', '#AF4BFF', '#FFA14B', 
                      '#4BFFEA', '#FF4BE5', '#FFF84B', '#4B4BFF', '#FF4B93']
        
        # History for potential merging and correction
        self.embedding_history = []
        self.recent_speaker_ids = []
        
        # For forced speaker changes
        self.last_speaker_id = None
        self.same_speaker_count = 0
        self.max_same_speaker_segments = 20  # Even more conservative - was 15
        
        # For new speaker creation control
        self.min_new_speaker_similarity = 0.40  # Increased from 0.35 - higher minimum similarity required
        self.short_utterance_threshold = 2.0  # Increased from 1.5s - longer utterances considered more reliable
        self.min_duration_for_new_speaker = 1.5  # Minimum duration required to create a new speaker
        
        # Tracking for first two speakers
        self.first_speaker_created = False
        self.second_speaker_created = False
        
        # Locks for thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Initialized incremental clustering with threshold {similarity_threshold}")
    
    def predict_speaker(self, embedding: np.ndarray, duration: float = 0.0) -> Tuple[str, float]:
        """Predict the speaker ID for a given embedding
        
        Args:
            embedding: Speaker embedding vector (normalized)
            duration: Duration of the speech segment
            
        Returns:
            Tuple of (speaker_id, confidence)
        """
        with self.lock:
            # For very short utterances, be more conservative
            is_short_utterance = duration < self.short_utterance_threshold
            
            # First, check if we have any existing speakers
            if not self.speakers:
                # Create the first speaker
                speaker_id = self._create_new_speaker(embedding, duration)
                self.first_speaker_created = True
                return speaker_id, 0.5  # Low initial confidence
            
            # Find the most similar speaker
            best_speaker_id, similarity = self._find_most_similar_speaker(embedding)
            
            # Get threshold adjusted by speaker confidence if enabled
            threshold = self.similarity_threshold
            if self.use_adaptive_threshold and best_speaker_id in self.speakers:
                speaker = self.speakers[best_speaker_id]
                # More embeddings = more reliable = can lower threshold
                confidence_factor = max(0.7, min(1.0, speaker.confidence))
                threshold = self.similarity_threshold * confidence_factor
            
            # For short utterances, increase the threshold to be more conservative
            if is_short_utterance:
                threshold *= 0.85  # Make it easier to match to existing speakers (was 0.9)
            
            # Check if we need to force a different speaker for variety
            force_new_speaker = False
            if best_speaker_id == self.last_speaker_id:
                self.same_speaker_count += 1
                # Only force speaker change for longer utterances and after many consecutive segments
                if self.same_speaker_count > self.max_same_speaker_segments and not is_short_utterance and duration > 3.0:
                    force_new_speaker = True
                    if self.debug:
                        logger.info(f"Forcing a new speaker after {self.same_speaker_count} segments from same speaker")
            else:
                self.same_speaker_count = 0
                
            # If similarity is high enough and we're not forcing a new speaker
            if similarity >= threshold and not force_new_speaker:
                speaker = self.speakers[best_speaker_id]
                speaker.add_embedding(embedding, duration)
                self.last_speaker_id = best_speaker_id
                
                if self.debug:
                    logger.info(f"Assigned to existing speaker {best_speaker_id} with similarity {similarity:.3f}")
                return best_speaker_id, similarity
            else:
                # For short utterances, be more cautious about creating new speakers
                if is_short_utterance or len(self.speakers) >= 3:
                    # For short utterances or if we already have 3+ speakers, prefer to assign 
                    # to the most similar existing speaker even if below threshold
                    if similarity >= self.min_new_speaker_similarity * 0.9:
                        speaker = self.speakers[best_speaker_id]
                        speaker.add_embedding(embedding, duration)
                        self.last_speaker_id = best_speaker_id
                        if self.debug:
                            logger.info(f"Conservative assignment to closest speaker {best_speaker_id} with below-threshold similarity {similarity:.3f}")
                        return best_speaker_id, similarity
                
                # If we have only one speaker so far, make it easier to create a second one, but still be cautious
                if len(self.speakers) == 1 and not self.second_speaker_created:
                    if duration > self.min_duration_for_new_speaker and similarity < 0.55:
                        speaker_id = self._create_new_speaker(embedding, duration)
                        self.last_speaker_id = speaker_id
                        self.second_speaker_created = True
                        if self.debug:
                            logger.info(f"Created second speaker {speaker_id} (similarity with first: {similarity:.3f})")
                        return speaker_id, 0.5
                    else:
                        # Otherwise assign to the first speaker
                        speaker = self.speakers[best_speaker_id]
                        speaker.add_embedding(embedding, duration)
                        self.last_speaker_id = best_speaker_id
                        if self.debug:
                            logger.info(f"Assigned to first speaker {best_speaker_id} (similarity: {similarity:.3f})")
                        return best_speaker_id, similarity
                
                # Handle forced speaker changes
                if force_new_speaker:
                    # When forcing change, use the next most similar speaker if possible
                    second_best_id, second_best_sim = self._find_second_best_speaker(embedding, best_speaker_id)
                    
                    if second_best_id and second_best_sim >= self.min_new_speaker_similarity * 0.8:
                        speaker = self.speakers[second_best_id]
                        speaker.add_embedding(embedding, duration)
                        self.last_speaker_id = second_best_id
                        self.same_speaker_count = 0
                        if self.debug:
                            logger.info(f"Forced change: assigned to second best speaker {second_best_id} with similarity {second_best_sim:.3f}")
                        return second_best_id, second_best_sim
                    
                    # If no good second best, just continue with the same speaker but reset the counter
                    speaker = self.speakers[best_speaker_id]
                    speaker.add_embedding(embedding, duration)
                    self.last_speaker_id = best_speaker_id
                    self.same_speaker_count = 0
                    if self.debug:
                        logger.info(f"Forced change attempted but no good alternative found, continuing with {best_speaker_id}")
                    return best_speaker_id, similarity
                
                # Create a new speaker only if similarity is quite low and duration is sufficient
                if similarity < self.min_new_speaker_similarity and duration > self.min_duration_for_new_speaker and len(self.speakers) < 3:
                    speaker_id = self._create_new_speaker(embedding, duration)
                    self.last_speaker_id = speaker_id
                    
                    if self.debug:
                        logger.info(f"Created new speaker {speaker_id} (best similarity was {similarity:.3f})")
                    
                    # Check if we need to merge speakers (could happen in background)
                    if len(self.speakers) > 2:
                        threading.Thread(target=self._check_for_speaker_merges).start()
                        
                    return speaker_id, 0.5  # Low initial confidence
                else:
                    # If similarity is not very low or duration is too short, assign to the closest speaker anyway
                    speaker = self.speakers[best_speaker_id]
                    speaker.add_embedding(embedding, duration)
                    self.last_speaker_id = best_speaker_id
                    if self.debug:
                        logger.info(f"Assigned to closest speaker {best_speaker_id} despite below-threshold similarity {similarity:.3f}")
                    return best_speaker_id, similarity
    
    def _find_most_similar_speaker(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Find the most similar speaker to the given embedding
        
        Args:
            embedding: Speaker embedding vector
            
        Returns:
            Tuple of (speaker_id, similarity)
        """
        max_similarity = -1
        best_speaker_id = None
        
        # Check similarity with all existing speakers
        for speaker_id, speaker in self.speakers.items():
            avg_embedding = speaker.get_average_embedding()
            if avg_embedding is None:
                continue
            
            # Compute cosine similarity
            similarity = self._compute_similarity(embedding, avg_embedding)
            
            # Apply recency weighting (favor recently active speakers)
            recency_factor = speaker.get_recency_weight()
            weighted_similarity = similarity * (0.7 + 0.3 * recency_factor)  # Increased recency influence
            
            if weighted_similarity > max_similarity:
                max_similarity = similarity  # Store actual similarity, not weighted
                best_speaker_id = speaker_id
                
            if self.debug:
                logger.debug(f"Similarity with speaker {speaker_id}: {similarity:.3f} (weighted: {weighted_similarity:.3f})")
        
        return best_speaker_id, max_similarity
    
    def _find_second_best_speaker(self, embedding: np.ndarray, exclude_id: str) -> Tuple[Optional[str], float]:
        """Find the second most similar speaker to the given embedding
        
        Args:
            embedding: Speaker embedding vector
            exclude_id: ID of speaker to exclude (usually the best match)
            
        Returns:
            Tuple of (speaker_id, similarity) or (None, 0) if no second best
        """
        max_similarity = -1
        best_speaker_id = None
        
        # Check similarity with all existing speakers except the excluded one
        for speaker_id, speaker in self.speakers.items():
            if speaker_id == exclude_id:
                continue
                
            avg_embedding = speaker.get_average_embedding()
            if avg_embedding is None:
                continue
            
            # Compute cosine similarity
            similarity = self._compute_similarity(embedding, avg_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_speaker_id = speaker_id
        
        if best_speaker_id is None:
            return None, 0
            
        return best_speaker_id, max_similarity
    
    def _create_new_speaker(self, embedding: np.ndarray, duration: float = 0.0) -> str:
        """Create a new speaker
        
        Args:
            embedding: Initial speaker embedding
            duration: Duration of the speech segment
            
        Returns:
            New speaker ID
        """
        # Generate a unique speaker ID
        speaker_id = f"S{self.next_speaker_id}"
        self.next_speaker_id += 1
        
        # Color assignment
        color = self.colors[len(self.speakers) % len(self.colors)]
        
        # Create new speaker object
        speaker = Speaker(
            id=speaker_id,
            name=f"Speaker {self.next_speaker_id}",
            color=color
        )
        speaker.add_embedding(embedding, duration)
        
        # Add to speakers dictionary
        self.speakers[speaker_id] = speaker
        
        # Check if we've exceeded the maximum number of speakers
        if len(self.speakers) > self.max_speakers:
            self._prune_least_active_speaker()
        
        logger.info(f"Created new speaker {speaker_id}")
        return speaker_id
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
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
    
    def _check_for_speaker_merges(self):
        """Check if any speakers should be merged (runs in background)"""
        with self.lock:
            if len(self.speakers) <= 1:
                return
                
            # Build a similarity matrix between all speakers
            speaker_ids = list(self.speakers.keys())
            embeddings = []
            
            for speaker_id in speaker_ids:
                speaker = self.speakers[speaker_id]
                if len(speaker.embeddings) < self.min_embeddings_for_reliability:
                    continue
                embeddings.append(speaker.get_average_embedding())
            
            if len(embeddings) <= 1:
                return
            
            # Convert to array for faster computation
            embeddings = np.array(embeddings)
            
            # Compute similarity matrix
            similarity_matrix = np.dot(embeddings, embeddings.T)
            
            # Find pairs above threshold (excluding self-similarity)
            merge_threshold = self.similarity_threshold * 0.9  # Slightly lower for merging
            
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > merge_threshold:
                        # Merge speakers
                        speaker1 = self.speakers[speaker_ids[i]]
                        speaker2 = self.speakers[speaker_ids[j]]
                        
                        # The one with more embeddings is likely more reliable
                        if len(speaker1.embeddings) >= len(speaker2.embeddings):
                            speaker1.merge_with(speaker2)
                            del self.speakers[speaker_ids[j]]
                            logger.info(f"Merged speaker {speaker_ids[j]} into {speaker_ids[i]}")
                        else:
                            speaker2.merge_with(speaker1)
                            del self.speakers[speaker_ids[i]]
                            logger.info(f"Merged speaker {speaker_ids[i]} into {speaker_ids[j]}")
                        
                        # Only do one merge at a time to avoid cascading effects
                        return
    
    def _prune_least_active_speaker(self):
        """Remove the least active speaker when we exceed the maximum"""
        with self.lock:
            min_activity = float('inf')
            least_active_id = None
            
            for speaker_id, speaker in self.speakers.items():
                # Measure of activity: recency and total duration
                activity_score = speaker.get_recency_weight() * speaker.total_duration
                
                if activity_score < min_activity:
                    min_activity = activity_score
                    least_active_id = speaker_id
            
            if least_active_id:
                del self.speakers[least_active_id]
                logger.info(f"Pruned least active speaker {least_active_id}")
    
    def get_num_speakers(self) -> int:
        """Get the number of speakers currently tracked"""
        return len(self.speakers)
    
    def get_speaker_names(self) -> Dict[str, str]:
        """Get a dictionary of speaker IDs to names"""
        return {id: speaker.name for id, speaker in self.speakers.items()}
    
    def rename_speaker(self, speaker_id: str, new_name: str) -> bool:
        """Rename a speaker
        
        Args:
            speaker_id: ID of the speaker to rename
            new_name: New name for the speaker
            
        Returns:
            Success flag
        """
        with self.lock:
            if speaker_id in self.speakers:
                self.speakers[speaker_id].name = new_name
                logger.info(f"Renamed speaker {speaker_id} to {new_name}")
                return True
            return False
    
    def get_speaker_stats(self) -> Dict[str, Dict]:
        """Get statistics for all speakers
        
        Returns:
            Dictionary of speaker IDs to their statistics
        """
        stats = {}
        for speaker_id, speaker in self.speakers.items():
            stats[speaker_id] = {
                "name": speaker.name,
                "color": speaker.color,
                "total_duration": speaker.total_duration,
                "speech_segments": speaker.speech_segments,
                "last_active_seconds_ago": time.time() - speaker.last_active,
                "confidence": speaker.confidence,
                "num_embeddings": len(speaker.embeddings)
            }
        return stats
    
    def plot_speaker_activity(self):
        """Return data for plotting speaker activity"""
        # This would be implemented to visualize speaker activity over time
        # For real-time visualization
        pass

# Example usage
if __name__ == "__main__":
    # Create a test clustering algorithm
    clustering = IncrementalClustering(similarity_threshold=0.75)
    
    # Generate some test embeddings
    speaker1_embs = [np.random.randn(192) for _ in range(5)]
    speaker2_embs = [np.random.randn(192) for _ in range(5)]
    
    # Normalize embeddings
    for i in range(len(speaker1_embs)):
        speaker1_embs[i] = speaker1_embs[i] / np.linalg.norm(speaker1_embs[i])
        speaker2_embs[i] = speaker2_embs[i] / np.linalg.norm(speaker2_embs[i])
    
    # Add slight correlation between embeddings from the same speaker
    corr_factor = 0.8
    base_emb1 = np.random.randn(192)
    base_emb2 = np.random.randn(192)
    base_emb1 = base_emb1 / np.linalg.norm(base_emb1)
    base_emb2 = base_emb2 / np.linalg.norm(base_emb2)
    
    for i in range(len(speaker1_embs)):
        speaker1_embs[i] = (corr_factor * base_emb1 + (1-corr_factor) * speaker1_embs[i])
        speaker1_embs[i] = speaker1_embs[i] / np.linalg.norm(speaker1_embs[i])
        
        speaker2_embs[i] = (corr_factor * base_emb2 + (1-corr_factor) * speaker2_embs[i])
        speaker2_embs[i] = speaker2_embs[i] / np.linalg.norm(speaker2_embs[i])
    
    # Test speaker identification
    print("Testing incremental clustering...")
    
    # Interleave embeddings to simulate a conversation
    for i in range(5):
        # Speaker 1
        speaker_id, confidence = clustering.predict_speaker(speaker1_embs[i], duration=2.0)
        print(f"Embedding {i} (Speaker 1) -> predicted {speaker_id} with confidence {confidence:.3f}")
        
        # Speaker 2
        speaker_id, confidence = clustering.predict_speaker(speaker2_embs[i], duration=2.0)
        print(f"Embedding {i} (Speaker 2) -> predicted {speaker_id} with confidence {confidence:.3f}")
    
    # Print final statistics
    print("\nFinal speaker statistics:")
    stats = clustering.get_speaker_stats()
    for speaker_id, speaker_stats in stats.items():
        print(f"Speaker {speaker_id}:")
        for key, value in speaker_stats.items():
            print(f"  {key}: {value}") 