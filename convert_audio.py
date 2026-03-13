import os
import librosa
import soundfile as sf

input_dataset = "dataset"
output_dataset = "processed_dataset"

classes = ["emergency", "normal"]

target_sr = 16000

for label in classes:

    input_folder = os.path.join(input_dataset, label)
    output_folder = os.path.join(output_dataset, label)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nChecking folder: {input_folder}")

    files = os.listdir(input_folder)

    print(f"Found files: {files}")

    for file in files:

        input_path = os.path.join(input_folder, file)

        print("Processing:", input_path)

        try:
            audio, sr = librosa.load(input_path, sr=16000, mono=True)

            output_file = os.path.splitext(file)[0] + ".wav"
            output_path = os.path.join(output_folder, output_file)

            sf.write(output_path, audio, target_sr)

        except Exception as e:
            print("Error:", e)