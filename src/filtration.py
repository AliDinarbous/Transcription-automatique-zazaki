import os
import pandas as pd
import evaluate
import torchaudio

# paths
input_csv = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/segment_csv/results.csv"
segments_dir = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/segment"
output_csv = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/segment_csv/final.csv"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# metric
cer_metric = evaluate.load("cer")

df = pd.read_csv(input_csv)

filtered_rows = []

for i, row in df.iterrows():
    clean = str(row["transcript_clean"])
    snr5 = str(row["transcript_snr_5"])
    snr10 = str(row["transcript_snr_10"])
    snr15 = str(row["transcript_snr_15"])

    # CER
    cer_5 = cer_metric.compute(predictions=[snr5], references=[clean])
    cer_10 = cer_metric.compute(predictions=[snr10], references=[clean])
    cer_15 = cer_metric.compute(predictions=[snr15], references=[clean])

    # diff
    cer_values = [cer_5, cer_10, cer_15]
    diff = max(cer_values) - min(cer_values)

    # durée audio
    path = os.path.join(segments_dir, row["wav_filename"])
    wav, sr = torchaudio.load(path)
    duration = wav.shape[1] / sr

    # score
    score = diff / max(duration, 1e-6)

    if score <= 0.07:
        filtered_rows.append({
            "wav_filename": row["wav_filename"],
            "transcript": clean
        })
final_df = pd.DataFrame(filtered_rows)
final_df.to_csv(output_csv, index=False)