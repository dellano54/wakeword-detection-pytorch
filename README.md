
# Wake Word Detection System

This repository contains an end-to-end implementation of a Wake Word Detection system. The system listens for a specific keyword (wake word) in audio streams, processes the audio, and triggers an action when the keyword is detected. It leverages deep learning techniques and audio feature extraction for accurate detection.

This model is very light and can be deployed on edge devices easily. The model weights is just **374 KB**

---

## Features

- **Real-time Wake Word Detection**: Utilizes `PyAudio` for live audio streaming and `torchaudio` for feature extraction.
- **Customizable Neural Network**: Implements a modular convolutional network (`StackedConvNet`) for learning wake word features.
- **Data Augmentation**: Enhances model training using techniques like noise addition, time shifting, and pitch shifting.
- **Efficient Training**: Includes scripts for training and validation with memory tracking.
- **Dataset Recording Tool**: A utility to record and organize wake word datasets.
- **Preprocessing and Feature Extraction**: Extracts MFCC features with configurable parameters.

---

## Directory Structure

```
.
├── engine.py                # Main wake word detection engine
├── model.py                # Defines the StackedConvNet model architecture
├── requirements.txt        # Modules necessary to make this program work
├── TRAIN/
│   ├── train.py             # Script for training the wake word model
│   ├── recorder.py          # Tool for recording wake word datasets
│   ├── model.py             # Duplicate of the main model script
│   ├── data-2seconds/       # Organized dataset
│       ├── background/
│       ├── crowd/
│       ├── random_speech/
│       ├── wakeword/
└── README.md                # Documentation (this file)
```

---

## Getting Started

### Prerequisites

- Python 3.8 or above
- Dependencies:
  - `torch`, `torchaudio`
  - `pyaudio`
  - `sounddevice`
  - `scipy`
  - `transformers`
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

---

## Usage

### 1. Dataset Preparation
Use the `recorder.py` script to record your wake word samples:
```bash
python TRAIN/recorder.py <n_times>
```
This will save audio samples in the `TRAIN/data-2seconds/` directory.

### 2. Training the Model
Train the wake word detection model using:
```bash
python TRAIN/train.py --dataset_folder TRAIN/data-2seconds --epochs 10 --device cuda
```

### 3. Running the Wake Word Detector
Start the wake word detection engine with:
```bash
python engine.py
```

---

## Model Architecture

The core model (`StackedConvNet`) uses:
- Stacked 1D convolutional layers with batch normalization and SiLU activation.
- PixelShuffle for efficient upsampling.
- Adaptive pooling for feature reduction.
- Fully connected layers for binary classification.

---

## Data Augmentation

During training, the following augmentations are applied:
- **Add Noise**: Random noise is added to the MFCC features.
- **Time Shift**: Shifts the waveform in time.
- **Pitch Shift**: Alters the pitch of the audio.



---

## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [Torchaudio](https://pytorch.org/audio/)
- [Sounddevice](https://python-sounddevice.readthedocs.io/)
