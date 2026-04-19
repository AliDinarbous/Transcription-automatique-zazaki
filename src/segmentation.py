import os
import torch
import librosa
import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad

# Paths
input_dir = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/audio_zazaki"
output_dir = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/segment"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = load_silero_vad()

segment_global_id = 0  # compteur_id

for file in os.listdir(input_dir):
    if not file.endswith(".m4a"):
        continue

    path = os.path.join(input_dir, file)

    # Load audio (16kHz)
    wav, sr = librosa.load(path, sr=16000)
    wav = torch.tensor(wav)

    # Segmentation
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=1000,
        max_speech_duration_s=15,
        threshold=0.5
    )

    # Save segments
    for segment in speech_timestamps:
        start, end = segment['start'], segment['end']
        segment_audio = wav[start:end].numpy()

        sf.write(f"{output_dir}/segment_{segment_global_id}.wav", segment_audio, 16000)
        segment_global_id += 1

