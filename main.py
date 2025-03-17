#!/usr/bin/env python3
"""
Real-Time Speaker Transcription and Diarization
Main entry point for the real-time diarization system
"""

import argparse
import logging
import sys
import time
from real_time_diarization import RealTimeDiarizer

def main():
    """Run the real-time speaker diarization system"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Real-time speaker transcription and diarization")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="base", 
        help="Whisper model to use (tiny, base, small, medium, large)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to run models on (cpu or cuda)"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.65, 
        help="Speaker similarity threshold. Lower values (0.55-0.65) detect more speakers but may be less accurate. "
             "Higher values (0.70-0.80) are more conservative about creating new speakers."
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging for detailed diagnostics and debugging information"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO
    if args.verbose:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("main")
    
    # Print welcome message
    print("\n" + "="*80)
    print("REAL-TIME SPEAKER TRANSCRIPTION AND DIARIZATION".center(80))
    print("="*80)
    print("\nThis system will transcribe spoken audio and identify different speakers in real-time.")
    print("Speak clearly into your microphone, and transcriptions will appear below.")
    
    print("\nSettings:")
    print(f"- Using Whisper model: {args.model}")
    print(f"- Device: {args.device}")
    print(f"- Speaker similarity threshold: {args.threshold}")
    print(f"- Verbose logging: {'Enabled' if args.verbose else 'Disabled'}")
    
    print("\nPress Ctrl+C to exit the program.")
    print("-"*80 + "\n")
    
    # Create and run the diarizer
    try:
        # Log the current threshold setting
        print(f"Running with speaker similarity threshold: {args.threshold} (lower = more speakers detected)")
        
        diarizer = RealTimeDiarizer(
            whisper_model=args.model,
            device=args.device,
            similarity_threshold=args.threshold
        )
        
        diarizer.run()
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Error in diarization: {e}", exc_info=True)
        print(f"Error: {e}")
        print("Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 