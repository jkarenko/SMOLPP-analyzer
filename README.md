# SMOLPP: Audio Similarity Analyzer

SMOLPP (Simple Music and Other Listenable Patterns Processor) is a Python-based command-line tool that analyzes audio tracks to estimate their similarity to a provided set of songs. It uses machine learning techniques to compute a similarity score ranging from 0 to 1, indicating how closely an input track matches the characteristics of the training set.

## Features

- Audio feature extraction using librosa
- Machine learning model built with PyTorch
- Similarity analysis of audio files
- Model saving and loading for reusability
- Comprehensive logging for debugging and tracking

## Installation

1. Ensure you have Python 3.11 installed.
2. Clone this repository:
   ```
   git clone https://github.com/yourusername/SMOLPP.git
   cd SMOLPP
   ```
3. Install the required dependencies using Poetry:
   ```
   poetry install
   ```

## Usage

SMOLPP can be used in several ways:

### Training a new model and analyzing a file

```
poetry run python smolpp.py input_file.mp3 training_set_dir
```

This will train a new model using the audio files in `training_set_dir` and then analyze the similarity of `input_file.mp3` to the training set.

### Training a model, saving it, and analyzing a file

```
poetry run python smolpp.py input_file.mp3 training_set_dir --save_model model.pth
```

This will train a new model, save it to `model.pth`, and then analyze `input_file.mp3`.

### Loading a pre-trained model and analyzing a file

```
poetry run python smolpp.py input_file.mp3 training_set_dir --load_model model.pth
```

This will load the pre-trained model from `model.pth` and use it to analyze `input_file.mp3`.

### Enabling debug logging

Add the `--debug` flag to any command to enable more verbose logging:

```
poetry run python smolpp.py input_file.mp3 training_set_dir --debug
```

## Example

Let's say you have a directory `pop_songs` containing a collection of pop music, and you want to see how similar a new song `new_track.mp3` is to this collection:

1. Train a model and analyze the new track:
   ```
   poetry run python smolpp.py new_track.mp3 pop_songs --save_model pop_model.pth
   ```
   This will output a similarity score and save the trained model.

2. Later, you can analyze another track using the same model:
   ```
   poetry run python smolpp.py another_track.mp3 pop_songs --load_model pop_model.pth
   ```

## Project Structure

- `smolpp.py`: Main script containing all the functionality
- `README.md`: This file
- `pyproject.toml`: Poetry configuration file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
