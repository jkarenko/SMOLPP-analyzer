import argparse
import logging
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    logger.info(f"Extracting features from {audio_file}")
    try:
        y, sr = librosa.load(audio_file)
    except Exception as e:
        logger.error(f"Error loading audio file {audio_file}: {str(e)}")
        raise ValueError(f"Error loading audio file {audio_file}: {str(e)}")

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    except Exception as e:
        logger.error(f"Error extracting features from {audio_file}: {str(e)}")
        raise ValueError(f"Error extracting features from {audio_file}: {str(e)}")

    features = {
        'tempo': tempo,
        'chroma_mean': np.mean(chroma),
        'mfcc_mean': np.mean(mfcc),
        'spectral_centroid_mean': np.mean(spectral_centroid)
    }

    logger.debug(f"Extracted features: {features}")
    return features


def train_model(training_data):
    if not training_data:
        logger.error("No training data provided")
        raise ValueError("No training data provided")

    logger.info(f"Starting feature extraction for {len(training_data)} files")
    features = []
    for file in training_data:
        try:
            features.append(extract_features(file))
        except ValueError as e:
            logger.warning(f"Skipping file {file}. {str(e)}")

    if not features:
        logger.warning("No valid features extracted from training data")
        # Instead of raising an exception, return a dummy model
        return SimilarityModel(4)  # 4 is the number of features we extract

    X = torch.tensor([list(f.values()) for f in features], dtype=torch.float32)

    logger.info(f"Initializing model with input size {X.shape[1]}")
    model = SimilarityModel(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    n_epochs = 100
    logger.info(f"Starting model training for {n_epochs} epochs")
    try:
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, torch.ones_like(outputs))
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
    except RuntimeError as e:
        logger.error(f"Error during model training: {str(e)}")
        raise ValueError(f"Error during model training: {str(e)}")

    logger.info("Model training completed")
    return model


def analyze_similarity(model, input_file):
    logger.info(f"Analyzing similarity for {input_file}")
    try:
        features = extract_features(input_file)
    except ValueError as e:
        logger.error(f"Error analyzing input file: {str(e)}")
        return 0.0  # Return 0 similarity if we can't extract features

    X = torch.tensor(list(features.values()), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        similarity = model(X).item()
    logger.info(f"Similarity score: {similarity:.4f}")
    return similarity


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
    parser.add_argument("input_file", help="Path to input audio file")
    parser.add_argument("training_set", help="Path to directory containing training audio files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save_model", help="Path to save the trained model")
    parser.add_argument("--load_model", help="Path to load a pre-trained model")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.debug("Starting SMOLPP")

    if not os.path.exists(args.input_file):
        logger.error(f"Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    if not args.load_model:
        if not os.path.isdir(args.training_set):
            logger.error(f"Training set directory '{args.training_set}' does not exist.")
            sys.exit(1)

        training_files = [os.path.join(args.training_set, f) for f in os.listdir(args.training_set) if
                          f.endswith(('.mp3', '.wav'))]

        if not training_files:
            logger.error(f"No .mp3 or .wav files found in the training set directory.")
            sys.exit(1)

    try:
        if args.load_model:
            model = load_model(args.load_model)
        else:
            logger.info(f"Training model with {len(training_files)} files")
            model = train_model(training_files)

            if args.save_model:
                save_model(model, model.fc1.in_features, args.save_model)

        similarity = analyze_similarity(model, args.input_file)
        print(f"Similarity score: {similarity:.4f}")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)

    logger.debug("SMOLPP completed successfully")


if __name__ == "__main__":
    main()
