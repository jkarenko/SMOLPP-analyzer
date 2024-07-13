Feature: SMOLPP Functionality
  As a user of SMOLPP
  I want to analyze audio similarity and manage models
  So that I can compare audio files effectively

  Background:
    Given I have a valid audio file "test_audio.mp3"
    And I have a directory "test_training_set" with audio files

  Scenario: Extracting features from a valid audio file
    When I extract features from the audio file
    Then I should get a dictionary of features
    And the dictionary should contain "tempo"
    And the dictionary should contain "chroma_mean"
    And the dictionary should contain "mfcc_mean"
    And the dictionary should contain "spectral_centroid_mean"

  Scenario: Training a model
    When I train a model using the training set
    Then I should get a trained model object

  Scenario: Saving and loading a model
    Given I have a trained model
    When I save and load the model
    Then the loaded model should have the same structure as the original model

  Scenario: Analyzing similarity
    Given I have a trained model
    When I analyze the similarity of "test_audio.mp3"
    Then I should get a similarity score between 0 and 1

  Scenario: End-to-end test
    When I run SMOLPP with "test_audio.mp3" and "test_training_set"
    Then it should output a similarity score
    And the process should complete without errors
