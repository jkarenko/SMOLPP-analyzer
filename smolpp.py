import argparse
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SimilarityModel(nn.Module):
    def __init__(self, input_size):
        super(SimilarityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file)
    except Exception as e:
        raise ValueError(f"Error loading audio file {audio_file}: {str(e)}")

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    except Exception as e:
        raise ValueError(f"Error extracting features from {audio_file}: {str(e)}")

    features = {
        'tempo': tempo,
        'chroma_mean': np.mean(chroma),
        'mfcc_mean': np.mean(mfcc),
        'spectral_centroid_mean': np.mean(spectral_centroid)
    }

    return features


def train_model(training_data):
    if not training_data:
        raise ValueError("No training data provided")

    features = []
    for file in training_data:
        try:
            features.append(extract_features(file))
        except ValueError as e:
            print(f"Warning: Skipping file {file}. {str(e)}")

    if not features:
        raise ValueError("No valid features extracted from training data")

    X = torch.tensor([list(f.values()) for f in features], dtype=torch.float32)

    model = SimilarityModel(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 100
    try:
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, torch.ones_like(outputs))
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
    except RuntimeError as e:
        raise ValueError(f"Error during model training: {str(e)}")

    return model


def analyze_similarity(model, input_file):
    try:
        features = extract_features(input_file)
    except ValueError as e:
        raise ValueError(f"Error analyzing input file: {str(e)}")

    X = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        similarity = model(X).item()
    return similarity


def main():
    parser = argparse.ArgumentParser(description="SMOLPP: Audio Similarity Analyzer")
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("training_set", help="Path to directory containing training audio files")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(args.training_set):
        print(f"Error: Training set directory '{args.training_set}' does not exist.")
        sys.exit(1)

    training_files = [os.path.join(args.training_set, f) for f in os.listdir(args.training_set) if
                      f.endswith(('.mp3', '.wav'))]

    if not training_files:
        print(f"Error: No .mp3 or .wav files found in the training set directory.")
        sys.exit(1)

    try:
        model = train_model(training_files)
        similarity = analyze_similarity(model, args.input_file)
        print(f"Similarity score: {similarity:.4f}")
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
