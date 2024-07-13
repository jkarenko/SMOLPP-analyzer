import argparse
import logging
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimilarityModel(nn.Module):
    def __init__(self, input_size):
        super(SimilarityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


def extract_features(audio_file):
    logger.info(f"Extracting features from {audio_file}")
    try:
        y, sr = librosa.load(audio_file)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        features = {
            'tempo': float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo),
            'chroma_mean': np.mean(chroma),
            'mfcc_mean': np.mean(mfcc),
            'spectral_centroid_mean': np.mean(spectral_centroid)
        }
        logger.debug(f"Extracted features: {features}")
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {audio_file}: {str(e)}")
        raise ValueError(f"Error extracting features from {audio_file}: {str(e)}")


def normalize_features(features):
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    return (features - min_vals) / (max_vals - min_vals)


def train_model(training_data):
    if not training_data:
        raise ValueError("No training data provided")

    logger.info(f"Starting feature extraction for {len(training_data)} files")
    features = []
    for file in training_data:
        try:
            file_features = extract_features(file)
            features.append(list(file_features.values()))
        except ValueError as e:
            logger.warning(f"Skipping file {file}. {str(e)}")

    if not features:
        logger.warning("No valid features extracted from training data")
        return SimilarityModel(4)

    features = np.array(features)
    normalized_features = normalize_features(features)

    indices = np.arange(normalized_features.shape[0])
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]

    x_train = torch.tensor(normalized_features[train_indices], dtype=torch.float32)
    x_val = torch.tensor(normalized_features[val_indices], dtype=torch.float32)

    logger.info(f"Initializing model with input size {x_train.shape[1]}")
    model = SimilarityModel(x_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    n_epochs = 1000
    logger.info(f"Starting model training for {n_epochs} epochs")
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(x_train)
        train_loss = criterion(train_outputs, torch.ones_like(train_outputs))
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, torch.ones_like(val_outputs))

        if (epoch + 1) % 100 == 0:
            logger.info(
                f'Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        if train_loss.item() < 0.01:
            logger.info(f"Stopping early at epoch {epoch + 1} due to low loss")
            break

    logger.info("Model training completed")
    return model


def analyze_similarity(model, input_file):
    logger.info(f"Analyzing similarity for {input_file}")
    try:
        features = extract_features(input_file)
        x = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            similarity = model(x).item()
        logger.info(f"Similarity score: {similarity:.4f}")
        return similarity
    except Exception as e:
        logger.error(f"Error analyzing input file: {str(e)}")
        return 0.0


def save_model(model, input_size, file_path):
    logger.info(f"Saving model to {file_path}")
    model_info = {
        'state_dict': model.state_dict(),
        'input_size': input_size
    }
    torch.save(model_info, file_path)


def load_model(file_path):
    logger.info(f"Loading model from {file_path}")
    model_info = torch.load(file_path)
    model = SimilarityModel(model_info['input_size'])
    model.load_state_dict(model_info['state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser(description="SMOLPP: Audio Similarity Analyzer")
    parser.add_argument("mode", choices=['train', 'analyze'], help="Mode of operation")
    parser.add_argument("--input_file", help="Path to input audio file (required for analyze mode)")
    parser.add_argument("--training_set",
                        help="Path to directory containing training audio files (required for train mode)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save_model", help="Path to save the trained model")
    parser.add_argument("--load_model", help="Path to load a pre-trained model")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.debug("Starting SMOLPP")

    if args.mode == 'analyze':
        if not args.input_file:
            logger.error("Input file is required for analyze mode.")
            sys.exit(1)
        if not args.load_model:
            logger.error("Pre-trained model is required for analyze mode.")
            sys.exit(1)

    if args.mode == 'train':
        if not args.training_set:
            logger.error("Training set is required for train mode.")
            sys.exit(1)
        if not os.path.isdir(args.training_set):
            logger.error(f"Training set directory '{args.training_set}' does not exist.")
            sys.exit(1)
        training_files = [os.path.join(args.training_set, f) for f in os.listdir(args.training_set) if
                          f.endswith(('.mp3', '.wav'))]
        if not training_files:
            logger.error(f"No .mp3 or .wav files found in the training set directory.")
            sys.exit(1)
        logger.info(f"Found {len(training_files)} audio files for training")

    try:
        if args.mode == 'train':
            logger.info(f"Training model with {len(training_files)} files")
            model = train_model(training_files)
            if args.save_model:
                save_model(model, model.fc1.in_features, args.save_model)
            print("Model training completed.")
        elif args.mode == 'analyze':
            model = load_model(args.load_model)
            similarity = analyze_similarity(model, args.input_file)
            print(f"Similarity score: {similarity:.4f}")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        sys.exit(1)

    logger.debug("SMOLPP completed successfully")


if __name__ == "__main__":
    main()
