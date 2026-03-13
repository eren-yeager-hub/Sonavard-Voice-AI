import os
import librosa
import soundfile as sf

input_dataset = "processed_dataset"
output_dataset = "segmented_dataset"

classes = ["emergency", "normal"]

sample_rate = 16000
segment_duration = 2  # seconds

segment_length = sample_rate * segment_duration

for label in classes:

    input_folder = os.path.join(input_dataset, label)
    output_folder = os.path.join(output_dataset, label)

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):

        file_path = os.path.join(input_folder, file)

        audio, sr = librosa.load(file_path, sr=sample_rate)

        total_segments = len(audio) // segment_length

        for i in range(total_segments):

            start = i * segment_length
            end = start + segment_length

            segment = audio[start:end]

            output_file = f"{os.path.splitext(file)[0]}_{i}.wav"

            output_path = os.path.join(output_folder, output_file)

            sf.write(output_path, segment, sample_rate)

        print("Segmented:", file)