# phoneme-informed-note-level-singing-transcription

A pretrained model for "A Phoneme-informed Neural Network Model for Note-level Singing Transcription", ICASSP 2023

The model will be released when the conference is over.

## Requirements

- torch==1.13.0
- torchaudio==0.13.0
- nnAudio==0.3.2
- mido==1.2.10
- mir_eval==0.7
- librosa==0.9.1
- numpy==1.23.4
- wquantiles==0.4

## Usage

```bash
$ python infer.py checkpoints/model.pt INPUT_FILE OUTPUT_FILE --bpm BPM_OF_INPUT_FILE --device DEVICE
```

## Blog with Demo Examples

https://seyong92.github.io/phoneme-informed-transcription-blog/
