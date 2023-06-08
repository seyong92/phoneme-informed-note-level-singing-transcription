# phoneme-informed-note-level-singing-transcription

A pretrained model for "A Phoneme-informed Neural Network Model for Note-level Singing Transcription", ICASSP 2023

## Requirements

- torch==1.13.1
- torchaudio==0.13.1
- nnAudio==0.3.2
- mido==1.2.10
- mir_eval==0.7
- librosa==0.9.1
- numpy==1.23.4
- wquantiles==0.4

Or if you are using [Poetry](https://python-poetry.org/), you can install the dependencies by running

```bash
$ poetry install
```

## Usage

```bash
$ python infer.py checkpoints/model.pt INPUT_FILE OUTPUT_FILE --bpm BPM_OF_INPUT_FILE --device DEVICE
```

- `INPUT_FILE` is the path to the input audio file.
- `OUTPUT_FILE` is the path to the output MIDI file. (If you do not give this argument, the default file name will be `out.mid`.)
- `BPM_OF_INPUT_FILE` is the BPM of the input audio file. (If you do not give this argument, the default value will be 120.)
- `DEVICE` is the device to run the model. (If you do not give this argument, the default device will be `cuda:0` if available, otherwise `cpu`.)

## Blog with Demo Examples

https://seyong92.github.io/phoneme-informed-transcription-blog/
