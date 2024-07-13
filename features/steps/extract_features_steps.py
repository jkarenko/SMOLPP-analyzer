import os

from behave import given, when, then

from smolpp import extract_features


@given('I have a valid audio file "{filename}"')
def step_impl(context, filename):
    context.audio_file = os.path.join('test_data', filename)
    assert os.path.exists(context.audio_file)


@when('I extract features from the audio file')
def step_impl(context):
    context.features = extract_features(context.audio_file)


@then('I should get a dictionary of features')
def step_impl(context):
    assert isinstance(context.features, dict)


@then('the dictionary should contain "{feature}"')
def step_impl(context, feature):
    assert feature in context.features
