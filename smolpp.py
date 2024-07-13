import argparse
import glob
import json
import logging
import os
import sys

import librosa
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
scipy.signal.hann = scipy.signal.windows.hann


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


def extract_features(audio_file, offset=0, duration=None):
    logger.info(f"Extracting features from {audio_file}")
    try:
        y, sr = librosa.load(audio_file, offset=offset, duration=duration)

        features = {'tempo': librosa.feature.tempo(y=y, sr=sr)[0],
                    'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y)),
                    'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
                    'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))}

        # Mel-frequency cepstral coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, coef in enumerate(mfcc):
            features[f'mfcc_{i}'] = np.mean(coef)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = np.mean(contrast)
        features['spectral_contrast_std'] = np.std(contrast)

        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)

        logger.debug(f"Extracted features: {features}")
        return features
    except Exception as e:
        logger.error(f"Error extracting features from {audio_file}: {str(e)}")
        raise ValueError(f"Error extracting features from {audio_file}: {str(e)}")


def calculate_metrics(y_true, y_pred_proba):
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred_proba)
    return accuracy, f1, auc


def normalize_features(features, min_vals, max_vals):
    return (features - min_vals) / (max_vals - min_vals + 1e-8)  # Add small epsilon to avoid division by zero


def extract_and_cache_features(directory):
    cache_file = os.path.join(directory, 'features.json')
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Corrupted cache file found in {directory}. Regenerating features.")
            os.remove(cache_file)
        except Exception as e:
            logger.warning(f"Error reading cache file in {directory}: {str(e)}. Regenerating features.")

    features = {}
    for file in glob.glob(os.path.join(directory, '*.mp3')):
        try:
            file_features = extract_features(file)
            # Convert numpy types to native Python types
            features[file] = [float(val) for val in file_features.values()]
        except ValueError as e:
            logger.warning(f"Skipping file {file}. {str(e)}")

    try:
        with open(cache_file, 'w') as f:
            json.dump(features, f)
    except Exception as e:
        logger.warning(f"Failed to write cache file in {directory}: {str(e)}")

    return features


def train_model(positive_dirs, negative_dirs, n_splits=5):
    features = []
    labels = []
    for label, directories in [(1, positive_dirs), (0, negative_dirs)]:
        for directory in directories:
            dir_features = extract_and_cache_features(directory)
            features.extend(dir_features.values())
            labels.extend([label] * len(dir_features))

    features = np.array(features)
    labels = np.array(labels)

    # Calculate normalization values
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)

    normalized_features = normalize_features(features, min_vals, max_vals)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_index, val_index) in enumerate(kf.split(normalized_features)):
        logger.info(f"Training fold {fold + 1}/{n_splits}")

        x_train = torch.tensor(normalized_features[train_index], dtype=torch.float32)
        y_train = torch.tensor(labels[train_index], dtype=torch.float32).unsqueeze(1)
        x_val = torch.tensor(normalized_features[val_index], dtype=torch.float32)
        y_val = torch.tensor(labels[val_index], dtype=torch.float32).unsqueeze(1)

        model = SimilarityModel(x_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

        best_val_loss = float('inf')
        best_val_metrics = None
        patience = 20
        epochs_without_improvement = 0

        n_epochs = 200
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            train_outputs = model(x_train)
            train_loss = criterion(train_outputs, y_train)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)

                # Calculate additional metrics
                val_accuracy, val_f1, val_auc = calculate_metrics(
                    y_val.numpy(), val_outputs.numpy()
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = (val_accuracy, val_f1, val_auc)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if (epoch + 1) % 50 == 0:
                logger.info(f'Fold {fold + 1}, Epoch [{epoch + 1}/{n_epochs}], '
                            f'Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                            f'Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')

        fold_results.append((best_val_loss, *best_val_metrics))

    # Log mean results across folds
    mean_loss, mean_accuracy, mean_f1, mean_auc = np.mean(fold_results, axis=0)
    logger.info(f"Cross-validation complete. "
                f"Mean validation - Loss: {mean_loss:.4f}, Accuracy: {mean_accuracy:.4f}, "
                f"F1: {mean_f1:.4f}, AUC: {mean_auc:.4f}")

    # Train final model on all data
    x_all = torch.tensor(normalized_features, dtype=torch.float32)
    y_all = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    final_model = SimilarityModel(x_all.shape[1])
    final_optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=0.01)

    # Final training loop
    for epoch in range(n_epochs):
        final_model.train()
        final_optimizer.zero_grad()
        outputs = final_model(x_all)
        loss = criterion(outputs, y_all)
        loss.backward()
        final_optimizer.step()

    logger.info("Final model training completed")
    return final_model, min_vals, max_vals


def analyze_similarity(model, min_vals, max_vals, input_file, offset=0, duration=None):
    logger.info(f"Analyzing similarity for {input_file}")
    try:
        features = extract_features(input_file, offset=offset, duration=duration)
        feature_values = np.array(list(features.values()))
        normalized_features = normalize_features(feature_values, min_vals, max_vals)
        x = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            similarity = model(x).item()
        logger.info(f"Similarity score: {similarity:.4f}")
        return similarity
    except Exception as e:
        logger.error(f"Error analyzing input file: {str(e)}")
        return 0.0


def save_model(model, input_size, min_vals, max_vals, file_path):
    logger.info(f"Saving model to {file_path}")
    model_info = {
        'state_dict': model.state_dict(),
        'input_size': input_size,
        'min_vals': min_vals,
        'max_vals': max_vals
    }
    torch.save(model_info, file_path)


def load_model(file_path):
    logger.info(f"Loading model from {file_path}")
    model_info = torch.load(file_path)
    model = SimilarityModel(model_info['input_size'])
    model.load_state_dict(model_info['state_dict'])
    return model, model_info['min_vals'], model_info['max_vals']


def main():
    parser = argparse.ArgumentParser(description="SMOLPP: Audio Similarity Analyzer")
    parser.add_argument("mode", choices=['train', 'analyze'], help="Mode of operation")
    parser.add_argument("--positive_dirs", nargs='+',
                        help="Paths to directories containing positive example audio files (required for train mode)")
    parser.add_argument("--negative_dirs", nargs='+',
                        help="Paths to directories containing negative example audio files (required for train mode)")
    parser.add_argument("--input_file",
                        help="Path to input audio file(s) (required for analyze mode, supports wildcards)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save_model", help="Path to save the trained model")
    parser.add_argument("--load_model", help="Path to load a pre-trained model")
    parser.add_argument("--offset", type=float, default=0, help="Start reading audio from this time (in seconds)")
    parser.add_argument("--duration", type=float, default=None, help="Only load up to this much audio (in seconds)")
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
        if not args.positive_dirs or not args.negative_dirs:
            logger.error("Both positive and negative directories are required for train mode.")
            sys.exit(1)
        for dir_list in [args.positive_dirs, args.negative_dirs]:
            for directory in dir_list:
                if not os.path.isdir(directory):
                    logger.error(f"Directory does not exist: {directory}")
                    sys.exit(1)
        positive_files = [f for d in args.positive_dirs for f in glob.glob(os.path.join(d, '*.mp3'))]
        negative_files = [f for d in args.negative_dirs for f in glob.glob(os.path.join(d, '*.mp3'))]
        if not positive_files or not negative_files:
            logger.error(f"No .mp3 files found in one or more of the specified directories.")
            sys.exit(1)
        logger.info(
            f"Found {len(positive_files)} positive examples and {len(negative_files)} negative examples for training")

    try:
        if args.mode == 'train':
            logger.info(f"Training model with positive and negative examples")
            model, min_vals, max_vals = train_model(args.positive_dirs, args.negative_dirs)
            if args.save_model:
                save_model(model, model.fc1.in_features, min_vals, max_vals, args.save_model)
            print("Model training completed.")
        elif args.mode == 'analyze':
            model, min_vals, max_vals = load_model(args.load_model)
            if '*' in args.input_file:
                input_files = glob.glob(os.path.expanduser(args.input_file))
            else:
                input_files = [os.path.expanduser(args.input_file)]

            if not input_files:
                logger.error(f"No files found matching the pattern: {args.input_file}")
                sys.exit(1)

            for file in input_files:
                if not os.path.exists(file):
                    logger.error(f"File not found: {file}")
                    continue
                similarity = analyze_similarity(model, min_vals, max_vals, file, offset=args.offset, duration=args.duration)
                print(f"File: {file}")
                print(f"Similarity score: {similarity * 100:.2f}%")
                print("---")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        sys.exit(1)

    logger.debug("SMOLPP completed successfully")


if __name__ == "__main__":
    main()
