import librosa
import soundfile as sf
import os

# Folder containing your MP3 recordings
input_folder = "my_recordings_mp3"

# Folder where converted files will go
output_folder = "segmented_dataset/emergency"

os.makedirs(output_folder, exist_ok=True)

target_sr = 16000
target_duration = 4
target_length = target_sr * target_duration

files = os.listdir(input_folder)

print("Files found:", files)

for file in files:

    if file.endswith(".mp3") or file.endswith(".mpeg"):

        input_path = os.path.join(input_folder, file)

        print("Processing:", input_path)

        audio, sr = librosa.load(input_path, sr=target_sr, mono=True)

        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            audio = librosa.util.fix_length(audio, size=target_length)

        output_file = os.path.splitext(file)[0] + ".wav"
        output_path = os.path.join(output_folder, output_file)

        sf.write(output_path, audio, target_sr)

        print("Saved:", output_path)

print("\nConversion finished.")