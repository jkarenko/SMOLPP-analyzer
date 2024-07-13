import os
import shutil
import tempfile

import torch
from behave import given, when, then

from smolpp import extract_features, train_model, save_model, load_model, analyze_similarity, SimilarityModel


@given('I have a valid audio file "{filename}"')
def step_impl(context, filename):
    context.audio_file = os.path.join('test_data', filename)
    assert os.path.exists(context.audio_file)


@given('I have a directory "{dirname}" with audio files')
def step_impl(context, dirname):
    context.training_set = tempfile.mkdtemp()
    for i in range(3):  # Create 3 dummy audio files
        with open(os.path.join(context.training_set, f'audio_{i}.mp3'), 'w') as f:
            f.write('dummy audio data')


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
    training_files = [os.path.join(context.training_set, f) for f in os.listdir(context.training_set)]
    context.model = train_model(training_files)


@then('I should get a trained model object')
def step_impl(context):
    assert isinstance(context.model, SimilarityModel)


@given('I have a trained model')
def step_impl(context):
    if not hasattr(context, 'model'):
        context.execute_steps('When I train a model using the training set')


@when('I save the model to "{filename}"')
def step_impl(context, filename):
    context.model_file = filename
    save_model(context.model, context.model.fc1.in_features, context.model_file)


@when('I load the model from "{filename}"')
def step_impl(context, filename):
    context.loaded_model = load_model(filename)


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


@when('I run SMOLPP with "{input_file}" and "{training_set}"')
def step_impl(context, input_file, training_set):
    input_path = os.path.join('test_data', input_file)
    context.output = os.popen(f'python smolpp.py {input_path} {context.training_set}').read()


@then('it should output a similarity score')
def step_impl(context):
    assert 'Similarity score:' in context.output


@then('the process should complete without errors')
def step_impl(context):
    assert 'Error:' not in context.output


def after_scenario(context, scenario):
    if hasattr(context, 'training_set'):
        shutil.rmtree(context.training_set)
    if hasattr(context, 'model_file') and os.path.exists(context.model_file):
        os.remove(context.model_file)
