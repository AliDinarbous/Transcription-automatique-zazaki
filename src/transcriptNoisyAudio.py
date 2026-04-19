import os
import torch
import librosa
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer

segments_dir = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/segment"
output_csv = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/segment_csv/results.csv"
model_dir = "/info/raid-etu/m1/s2506992/projet-m1-asr/results/whisperBaseKurdZazakiV3/checkpoint-200"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = WhisperTokenizer.from_pretrained(model_dir)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
processor = WhisperProcessor(feature_extractor, tokenizer)

model = WhisperForConditionalGeneration.from_pretrained(model_dir)
model.to(device)
model.eval()


def add_noise(wav, snr_db):
    wav = wav / (wav.abs().max() + 1e-8)
    noise = torch.randn_like(wav)

    signal_power = wav.pow(2).mean()
    noise_power = noise.pow(2).mean()

    snr = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (snr * noise_power))

    return wav + scale * noise


def transcribe(wav):
    inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


results = []

files = sorted([f for f in os.listdir(segments_dir) if f.endswith(".wav")])

for i, file in enumerate(files):
    path = os.path.join(segments_dir, file)

    # load audio
    wav, _ = librosa.load(path, sr=16000)
    wav = torch.tensor(wav)

    # clean
    transcript_clean = transcribe(wav)

    # noisy
    transcript_snr_5 = transcribe(add_noise(wav, 5))
    transcript_snr_10 = transcribe(add_noise(wav, 10))
    transcript_snr_15 = transcribe(add_noise(wav, 15))

    results.append({
        "wav_filename": file,
        "transcript_clean": transcript_clean,
        "transcript_snr_5": transcript_snr_5,
        "transcript_snr_10": transcript_snr_10,
        "transcript_snr_15": transcript_snr_15
    })

    if i % 100 == 0:
        print(f"Processed {i}/{len(files)}")

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)

print("Done ")