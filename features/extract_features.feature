Feature: Audio Feature Extraction
  As a user of SMOLPP
  I want to extract features from audio files
  So that I can analyze their similarity

  Scenario: Extracting features from a valid audio file
    Given I have a valid audio file "test_audio.mp3"
    When I extract features from the audio file
    Then I should get a dictionary of features
    And the dictionary should contain "tempo"
    And the dictionary should contain "chroma_mean"
    And the dictionary should contain "mfcc_mean"
    And the dictionary should contain "spectral_centroid_mean"
