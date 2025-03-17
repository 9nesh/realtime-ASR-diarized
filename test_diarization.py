#!/usr/bin/env python3
"""
Test Script for Real-Time Speaker Diarization
This script tests the fixed diarization system with two different speakers.
"""

import time
import sys
import logging
import argparse
from real_time_diarization import RealTimeDiarizer

def main():
    """Run the test script for real-time diarization"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test the real-time speaker diarization system")
    parser.add_argument("--model", type=str, default="base", help="Whisper model to use (tiny, base, small, medium, large)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run models on (cpu or cuda)")
    parser.add_argument("--threshold", type=float, default=0.65, 
                        help="Speaker similarity threshold. Lower values (0.55-0.65) detect more speakers but may be less accurate. "
                             "Higher values (0.70-0.80) are more conservative about creating new speakers.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Print test instructions
    print("\n" + "="*80)
    print("REAL-TIME SPEAKER DIARIZATION TEST")
    print("="*80)
    print("\nThis test will run the diarization system for 60 seconds.")
    print("During this time, please speak into your microphone, alternating between two speakers.")
    print("The system will attempt to identify when speaker changes occur.")
    print("\nTIPS:")
    print("- Speak clearly and at a normal volume")
    print("- Alternate between speakers every 5-10 seconds")
    print("- Leave a brief pause between speaker changes")
    print("- Speak sentences, not just single words")
    print("\nCurrent settings:")
    print(f"- Using Whisper model: {args.model}")
    print(f"- Device: {args.device}")
    print(f"- Speaker similarity threshold: {args.threshold}")
    print(f"- Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    print("\nPress Enter to start the test (will run for 60 seconds)...")
    input()
    
    # Create and run the diarizer
    try:
        print("\nStarting diarization test...")
        diarizer = RealTimeDiarizer(
            whisper_model=args.model,
            device=args.device,
            similarity_threshold=args.threshold
        )
        
        # Start diarization in a separate thread
        import threading
        
        stop_event = threading.Event()
        
        def run_diarizer():
            try:
                diarizer.start_audio_capture()
                diarizer.process_audio()
            except Exception as e:
                print(f"Error in diarization thread: {e}")
                stop_event.set()
        
        # Start the thread
        diarizer_thread = threading.Thread(target=run_diarizer)
        diarizer_thread.daemon = True
        diarizer_thread.start()
        
        # Run for 60 seconds
        print("\nTest running. Please speak into the microphone...")
        print("Two or more speakers should take turns speaking...")
        print("The test will automatically end after 60 seconds.\n")
        
        start_time = time.time()
        try:
            while time.time() - start_time < 60 and not stop_event.is_set():
                # Display a countdown
                remaining = 60 - int(time.time() - start_time)
                if remaining % 10 == 0 and remaining > 0:
                    print(f"Remaining time: {remaining} seconds")
                time.sleep(1)
        except KeyboardInterrupt:
            print("Test interrupted by user.")
        
        # Stop the diarizer
        print("\nTest completed. Stopping diarization...")
        diarizer.stop()
        
        # Print test summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total audio chunks processed: {diarizer.processed_chunks}")
        print(f"Chunks with speech detected: {diarizer.detected_speech_chunks}")
        print(f"Number of speakers detected: {diarizer.clustering.get_num_speakers()}")
        
        # Show statistics for each speaker
        speaker_stats = diarizer.clustering.get_speaker_stats()
        print("\nSpeaker Statistics:")
        for speaker_id, stats in speaker_stats.items():
            print(f"- {stats['name']}: {stats['speech_segments']} segments, {stats['total_duration']:.2f}s total")
        
        print("\nThank you for testing the diarization system!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        raise

if __name__ == "__main__":
    main() 