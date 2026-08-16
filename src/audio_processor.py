import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
from pathlib import Path

def preprocess_audio(input_path, output_path):
    # 1. Load audio (Librosa converts almost anything to a float array)
    # sr=None keeps the original sampling rate
    y, sr = librosa.load(input_path, sr=40000) 

    # 2. Denoise (ML magic: it estimates the noise floor and subtracts it)
    # We take a small snippet of the beginning to 'profile' the noise
    print((len(y), sr))
    reduced_noise = nr.reduce_noise(y=y, sr=sr)

    # 3. Normalize (Make the loudest peak 1.0)
    normalized_y = librosa.util.normalize(reduced_noise)

    # 4. Trim Silence (top_db=20 is quite sensitive)
    trimmed_y, _ = librosa.effects.trim(normalized_y, top_db=25)


    directory_path = Path('/'.join(output_path.split('/')[:-1]) + '/')
    directory_path.mkdir(parents=True, exist_ok=True)
    
    # 5. Save as high-quality WAV
    sf.write(output_path, trimmed_y, sr, subtype='PCM_16')
    print(f"Processed: {input_path} -> {output_path}")

# --- TEST IT ---
# 1. Record a 5-sec voice message with background noise (fan, TV)
# 2. Run this function on it.

def main():
    preprocess_audio('data/train/samples/sample1.ogg', 'data/train/output/sample1.wav')
    preprocess_audio('data/train/samples/sample2.ogg', 'data/train/output/sample2.wav')

if __name__ == '__main__':
    main()