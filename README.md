# Wake Word Detection Model

This project is a **Wake Word Detection Model** built using **TensorFlow**. It listens for a specific phrase (wake word) and triggers actions, making it useful for various applications such as voice assistants and security systems.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)

## Introduction

The Wake Word Detection Model listens for a specific wake word (such as "Hey Safy") and triggers an action when detected. This model is designed to be robust across different background noises and speaker variations. The project leverages **TensorFlow** for the wake word classification and detection task.

## Project Structure

```bash
.
├── PreProcessing.py        # Preprocesses the raw audio data for training
├── RunParallely.py         # Runs predictions in parallel on new audio data
├── PreparingData.py        # Prepares and handles datasets
├── audio_data/             # Directory containing raw audio data
├── background_sound/       # Directory containing background noise
├── final_audio_data_csv    # Metadata for processed audio files
├── plot_cm.py              # Script to plot confusion matrix
├── prediction.py           # Script to make predictions on new data
├── prediction.wav          # Sample audio file for testing
├── req.txt                 # Requirements file for Python dependencies
├── saved_model/            # Directory containing the trained model
├── training.py             # Script for training the TensorFlow model
└── venv/                   # Virtual environment directory
```


## Installation
```
git clone <repository_link>
cd <repository_name>

python3 -m venv venv
source venv/bin/activate  # for macOS/Linux
venv\Scripts\activate  # for Windows

pip install -r req.txt
```

## Dataset Preparation
Audio Data: Place your raw wake word and non-wake word audio samples in the audio_data/ directory.
Background Sound: Place background noise files in the background_sound/ directory.

Run pre-processing
```
python PreparingData.py
python PreProcessing.py
```

## Training the Model
```
python training.py
```

## Making Predictions

Single Prediction:
```
python prediction.py
```
Parallel Predictions: To run predictions in parallel:
```python RunParallely.py```

