import argparse
import glob
import json
import logging
import os
import sys
import tempfile
import hashlib
import signal
import warnings
from contextlib import contextmanager

import librosa
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import yt_dlp
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
scipy.signal.hann = scipy.signal.windows.hann
warnings.filterwarnings("ignore", message="The MPEG_LAYER_III subtype is unknown to TorchAudio.*")


class SimilarityModel(nn.Module):
    def __init__(self, input_size):
        super(SimilarityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


@contextmanager
def youtube_audio(url):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_template,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            audio_path = f'{os.path.splitext(filename)[0]}.mp3'
            if os.path.exists(audio_path):
                yield audio_path
            else:
                raise FileNotFoundError(f"Downloaded audio file not found: {audio_path}")


def extract_features(audio_file, use_gpu=False, chunk_duration=10):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

    # Load audio metadata
    metadata = torchaudio.info(audio_file)
    sample_rate = metadata.sample_rate
    num_frames = metadata.num_frames
    duration = num_frames / sample_rate

    # Set up spectrogram parameters
    n_fft = 2048
    win_length = None
    hop_length = 512
    n_mels = 128
    n_mfcc = 13

    # Initialize transforms
    spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length).to(device)
    mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length).to(
        device)
    mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
                            melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'hop_length': hop_length}).to(device)
    spectral_centroid = T.SpectralCentroid(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length).to(device)

    # Initialize feature accumulators
    features = {
        'spectrogram_mean': 0,
        'spectrogram_std': 0,
        'melspectrogram_mean': 0,
        'melspectrogram_std': 0,
        'spectral_centroid_mean': 0,
        'spectral_centroid_std': 0,
    }
    for i in range(n_mfcc):
        features[f'mfcc_{i}_mean'] = 0
        features[f'mfcc_{i}_std'] = 0

    # Process audio in chunks
    chunk_size = int(chunk_duration * sample_rate)
    num_chunks = int(np.ceil(num_frames / chunk_size))

    for i in tqdm(range(num_chunks), desc=f"Processing {os.path.basename(audio_file)}"):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, num_frames)

        waveform, _ = torchaudio.load(audio_file, frame_offset=start_frame, num_frames=end_frame - start_frame)
        waveform = waveform.to(device)

        # Skip processing if the chunk is silent
        if torch.allclose(waveform, torch.zeros_like(waveform)):
            logger.debug(f"Skipping silent chunk {i}")
            continue

        # Extract features for this chunk
        spec = spectrogram(waveform)
        mel_spec = mel_spectrogram(waveform)
        mfcc = mfcc_transform(waveform)

        # Spectral centroid calculation with improved error handling
        try:
            centroid = spectral_centroid(waveform)
            # Replace NaN and Inf values with zeros
            centroid = torch.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.error(f"Error calculating spectral centroid for chunk {i}: {str(e)}")
            centroid = torch.zeros_like(waveform)

        # Update accumulators
        features['spectrogram_mean'] += spec.mean().item()
        features['spectrogram_std'] += spec.std().item()
        features['melspectrogram_mean'] += mel_spec.mean().item()
        features['melspectrogram_std'] += mel_spec.std().item()
        features['spectral_centroid_mean'] += centroid.mean().item()
        features['spectral_centroid_std'] += centroid.std().item()

        for j in range(n_mfcc):
            features[f'mfcc_{j}_mean'] += mfcc[0, j].mean().item()
            features[f'mfcc_{j}_std'] += mfcc[0, j].std().item()

    # Compute final average
    for key in features:
        features[key] /= num_chunks

    return features


def calculate_metrics(y_true, y_pred_proba):
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred_proba)
    return accuracy, f1, auc


def normalize_features(features, min_vals, max_vals):
    return (features - min_vals) / (max_vals - min_vals + 1e-8)  # Add small epsilon to avoid division by zero


def cached_extract_features(audio_file, use_gpu=False, chunk_duration=10):
    # Create a unique identifier for this file and extraction parameters
    file_hash = hashlib.md5(open(audio_file, 'rb').read()).hexdigest()
    cache_key = f"{file_hash}_{use_gpu}_{chunk_duration}"
    cache_dir = os.path.join(os.path.dirname(audio_file), '.feature_cache')
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")

    # Check if cached features exist
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Extract features
    features = extract_features(audio_file, use_gpu, chunk_duration)

    # Cache the results
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(features, f)

    return features


def extract_file_features(file, use_gpu):
    return extract_features(file, use_gpu=use_gpu)


def extract_features_for_directory(directory, use_gpu=False):
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

    files = glob.glob(os.path.join(directory, '*.mp3'))

    with ProcessPoolExecutor() as executor:
        features = list(executor.map(extract_file_features, files, [use_gpu] * len(files)))

    feature_dict = dict(zip(files, features))

    try:
        with open(cache_file, 'w') as f:
            json.dump(feature_dict, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to write cache file in {directory}: {str(e)}")

    return feature_dict


def extract_and_cache_features(directory, use_gpu=False):
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
            file_features = extract_features(file, use_gpu=use_gpu)
            # Convert numpy types to native Python types
            features[file] = {k: float(v) for k, v in file_features.items()}
        except Exception as e:
            logger.warning(f"Skipping file {file}. {str(e)}")

    try:
        with open(cache_file, 'w') as f:
            json.dump(features, f)
    except Exception as e:
        logger.warning(f"Failed to write cache file in {directory}: {str(e)}")

    return features


def train_model(positive_dirs, negative_dirs, n_splits=5, use_gpu=False):
    features = []
    labels = []
    for label, directories in [(1, positive_dirs), (0, negative_dirs)]:
        for directory in directories:
            dir_features = extract_and_cache_features(directory, use_gpu)
            features.extend(dir_features.values())
            labels.extend([label] * len(dir_features))

    # Extract keys from the first feature dictionary
    feature_keys = list(features[0].keys())

    # Convert features to a list of lists containing only values
    feature_values = [list(feature.values()) for feature in features]

    # Convert to numpy array
    features = np.array(feature_values)
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


def analyze_similarity(model, min_vals, max_vals, input_file, is_youtube_url=False, use_gpu=False):
    logger.debug(f"Analyzing similarity for {'YouTube video' if is_youtube_url else 'file'}: {input_file}")
    try:
        if is_youtube_url:
            with youtube_audio(input_file) as audio_file:
                logger.debug(f"Downloaded YouTube audio to: {audio_file}")
                return analyze_audio_file(model, min_vals, max_vals, audio_file, use_gpu)
        else:
            return analyze_audio_file(model, min_vals, max_vals, input_file, use_gpu)
    except Exception as e:
        logger.error(f"Error analyzing input: {str(e)}")
        return 0.0


def analyze_audio_file(model, min_vals, max_vals, audio_file, use_gpu=False):
    features = extract_features(audio_file, use_gpu=use_gpu)
    feature_values = np.array(list(features.values()))
    normalized_features = normalize_features(feature_values, min_vals, max_vals)
    x = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        similarity = model(x).item()
    logger.debug(f"Similarity score: {similarity:.4f}")
    return similarity


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
    logger.debug(f"Loading model from {file_path}")
    model_info = torch.load(file_path)
    model = SimilarityModel(model_info['input_size'])
    model.load_state_dict(model_info['state_dict'])
    return model, model_info['min_vals'], model_info['max_vals']


def signal_handler(signum, frame):
    print("\nInterrupt received, stopping...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="SMOLPP: Audio Similarity Analyzer")
    parser.add_argument("mode", choices=['train', 'predict', 'extract'], help="Mode of operation")
    parser.add_argument("--dirs", nargs='+',
                        help="Paths to directories containing audio files (required for extract mode)")
    parser.add_argument("--positive_dirs", nargs='+',
                        help="Paths to directories containing positive example audio files (required for train mode)")
    parser.add_argument("--negative_dirs", nargs='+',
                        help="Paths to directories containing negative example audio files (required for train mode)")
    parser.add_argument("--input_file",
                        help="Path to input audio file(s) (required for predict mode, supports wildcards)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save_model", help="Path to save the trained model")
    parser.add_argument("--load_model", help="Path to load a pre-trained model")
    parser.add_argument("--yt-dlp", action="store_true", help="Treat input as URL and download audio using yt-dlp")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU acceleration for feature extraction")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Starting SMOLPP")

    use_gpu = args.use_gpu and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.init()
        logger.info("GPU acceleration enabled")
    else:
        logger.warning("GPU acceleration not available or not enabled")

    if args.mode == 'extract':
        if not args.dirs:
            logger.error("At least one directory is required for extract mode.")
            sys.exit(1)
        for directory in args.dirs:
            if not os.path.isdir(directory):
                logger.error(f"Directory does not exist: {directory}")
                sys.exit(1)
            extract_features_for_directory(directory, use_gpu=use_gpu)
        logger.info("Feature extraction completed for all specified directories.")
        sys.exit(0)

    if args.mode == 'predict':
        if not args.input_file:
            logger.error("Input file is required for predict mode.")
            sys.exit(1)
        if not args.load_model:
            logger.error("Pre-trained model is required for predict mode.")
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
        if args.mode == 'extract':
            for directory in args.dirs:
                if not os.path.isdir(directory):
                    logger.error(f"Directory does not exist: {directory}")
                    continue
                files = glob.glob(os.path.join(directory, '*.mp3'))
                for file in tqdm(files, desc=f"Extracting features from {directory}"):
                    try:
                        extract_file_features(file, args.use_gpu)
                    except Exception as e:
                        logger.error(f"Error processing {file}: {str(e)}")
            logger.info("Feature extraction completed for all specified directories.")

        elif args.mode == 'train':
            logger.info("Training model with positive and negative examples")
            model, min_vals, max_vals = train_model(args.positive_dirs, args.negative_dirs, use_gpu=args.use_gpu)
            if args.save_model:
                save_model(model, model.fc1.in_features, min_vals, max_vals, args.save_model)
            logger.info("Model training completed.")

        elif args.mode == 'predict':
            model, min_vals, max_vals = load_model(args.load_model)
            if args.yt_dlp:
                if not args.input_file.startswith('http'):
                    logger.error("Invalid YouTube URL")
                    sys.exit(1)
                input_files = [args.input_file]
            elif '*' in args.input_file:
                input_files = glob.glob(os.path.expanduser(args.input_file))
            else:
                input_files = [os.path.expanduser(args.input_file)]

            if not input_files:
                logger.error(f"No {'URLs' if args.yt_dlp else 'files'} found matching the input")
                sys.exit(1)

            for file in input_files:
                if not args.yt_dlp and not os.path.exists(file):
                    logger.error(f"File not found: {file}")
                    continue
                similarity = analyze_similarity(model, min_vals, max_vals, file,
                                                is_youtube_url=args.yt_dlp, use_gpu=args.use_gpu)
                logger.info(f"{file} similarity: {similarity * 100:.2f}%")
    except KeyboardInterrupt:
        print("\nInterrupt received, stopping...")
    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
    finally:
        logger.info("SMOLPP completed")


if __name__ == "__main__":
    main()
