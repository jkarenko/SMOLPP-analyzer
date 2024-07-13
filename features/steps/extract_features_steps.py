import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from behave import given, when, then

from smolpp import extract_features, train_model, analyze_similarity, SimilarityModel, save_model, load_model


@given('I have a valid audio file "{filename}"')
def step_impl(context, filename):
    context.audio_file = os.path.join('test_data', filename)
    if not os.path.exists(context.audio_file):
        sample_rate = 44100
        duration = 3
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(440 * 2 * np.pi * t)
        sf.write(context.audio_file, audio_data, sample_rate, format='wav')
    context.model_path = os.path.join(tempfile.gettempdir(), 'test_model.pth')


@given('I have directories with positive and negative audio files')
def step_impl(context):
    context.positive_dir = tempfile.mkdtemp()
    context.negative_dir = tempfile.mkdtemp()
    for i, directory in enumerate([context.positive_dir, context.negative_dir]):
        for j in range(3):
            filename = os.path.join(directory, f'audio_{j}.wav')
            sample_rate = 44100
            duration = 2
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin((440 + i * 100 + j * 50) * 2 * np.pi * t)
            sf.write(filename, audio_data, sample_rate, format='wav')


@when('I extract features from the audio file')
def step_impl(context):
    context.features = extract_features(context.audio_file)


@then('I should get a dictionary of features')
def step_impl(context):
    assert isinstance(context.features, dict)


@then('the dictionary should contain "{feature}"')
def step_impl(context, feature):
    assert feature in context.features


@when('I train a model using the training set')
def step_impl(context):
    context.model = train_model([context.positive_dir], [context.negative_dir])
    save_model(context.model, context.model.fc1.in_features, context.model_path)


@then('I should get a trained model object')
def step_impl(context):
    assert isinstance(context.model, SimilarityModel)


@given('I have a trained model')
def step_impl(context):
    if not hasattr(context, 'model'):
        context.execute_steps('When I train a model using the training set')


@when('I save and load the model')
def step_impl(context):
    context.loaded_model = load_model(context.model_path)


@then('the loaded model should have the same structure as the original model')
def step_impl(context):
    assert isinstance(context.loaded_model, SimilarityModel)
    assert context.loaded_model.fc1.in_features == context.model.fc1.in_features
    assert torch.all(context.loaded_model.fc1.weight == context.model.fc1.weight)


@when('I analyze the similarity of "{filename}"')
def step_impl(context, filename):
    context.similarity_score = analyze_similarity(context.model, os.path.join('test_data', filename))


@then('I should get a similarity score between 0 and 1')
def step_impl(context):
    assert 0 <= context.similarity_score <= 1


@when('I run SMOLPP with "{input_file}" in train mode')
def step_impl(context, input_file):
    input_path = os.path.join('test_data', input_file)
    context.output = os.popen(
        f'python smolpp.py train --positive_dirs {context.positive_dir} --negative_dirs {context.negative_dir} --save_model {context.model_path}').read()


@when('I run SMOLPP with "{input_file}" in analyze mode')
def step_impl(context, input_file):
    input_path = os.path.join('test_data', input_file)
    context.output = os.popen(
        f'python smolpp.py analyze --input_file {input_path} --load_model {context.model_path}').read()


@then('it should output a similarity score')
def step_impl(context):
    assert 'Similarity score:' in context.output


@then('the process should complete without errors')
def step_impl(context):
    assert 'Error:' not in context.output
