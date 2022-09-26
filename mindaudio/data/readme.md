# mindaudio.data

## [io](/mindaudio/data/io)
- read
- write

## [features](/mindaudio/data/features)
- stft
- istft
- ...

## [augment](/mindaudio/data/augment)
- speed_perturb
- spec_aug

## [masks](/mindaudio/data/masks)

## [collate](/mindaudio/data/collate)

## [datasets](/mindaudio/data/datasets)
- LJSpeechTTS
- LibriSpeechASR
- LibriTTSASR

## Example usage of making a dataset for text-to-speech training:

see [preprocess ljspeech](/mindaudio/examples/tacotron2/preprocess_tacotron2_ljspeech.py) and [train tts with ljspeech](/mindaudio/examples/tacotron2/dataset.py)