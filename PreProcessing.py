import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define supported audio formats
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

#### LOADING THE VOICE DATA FOR VISUALIZATION ###
walley_sample = "background_sounds/99.wav"
data, sample_rate = librosa.load(walley_sample)

##### VISUALIZING WAVE FORM ##
plt.title("Wave Form")
librosa.display.waveshow(data, sr=sample_rate)
plt.show()

##### VISUALIZING MFCC #######
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print("Shape of mfcc:", mfccs.shape)

plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.show()

##### Doing this for every sample ##

all_data = []

data_path_dict = {
    0: ["background_sounds/" + file_path for file_path in os.listdir("background_sounds/")],
    1: ["audio_data/" + file_path for file_path in os.listdir("audio_data/")]
}

# Filter and process only audio files
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        # Skip files that don't have an audio extension
        if not single_file.lower().endswith(AUDIO_EXTENSIONS):
            print(f"Skipping non-audio file: {single_file}")
            continue
        try:
            audio, sample_rate = librosa.load(single_file)  # Loading file
            mfcc = librosa.feature.mfcc(
                y=audio, sr=sample_rate, n_mfcc=40)  # Applying mfcc
            mfcc_processed = np.mean(mfcc.T, axis=0)  # Some pre-processing
            all_data.append([mfcc_processed, class_label])
        except Exception as e:
            print(f"Error processing {single_file}: {e}")
    print(f"Info: Successfully Preprocessed Class Label {class_label}")

df = pd.DataFrame(all_data, columns=["feature", "class_label"])

###### SAVING FOR FUTURE USE ###
df.to_pickle("final_audio_data_csv/audio_data.csv")
