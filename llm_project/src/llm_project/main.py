import os
import librosa
import numpy as np
from pathlib import Path

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Points to llm_project/
DATA_DIR = PROJECT_ROOT / "data"
SONGS_DIR = DATA_DIR / "songs"

# Ensure directories exist
SONGS_DIR.mkdir(parents=True, exist_ok=True)

def get_audio_embedding(audio_path):
    """Extract MFCC features and return the mean embedding."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate mean embedding
        embedding = np.mean(mfcc, axis=1)
        return embedding
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    audio_file = "release.mp3"  # Place your audio file in the songs directory
    audio_path = SONGS_DIR / audio_file
    
    if not audio_path.exists():
        print(f"Audio file not found at {audio_path}")
        print(f"Please place your audio file in: {SONGS_DIR}")
    else:
        embedding = get_audio_embedding(audio_path)
        if embedding is not None:
            print(f"Successfully processed: {audio_path}")
            print("Embedding shape:", embedding.shape)
            print("Embedding vector:", embedding)
