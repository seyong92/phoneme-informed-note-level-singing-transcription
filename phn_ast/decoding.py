from torch import nn
import numpy as np
import torch
import wquantiles
import librosa


class FramewiseDecoder:
    def __init__(self, config):
        self.sr = config['sample_rate']

        self.win_length = config['win_length']
        self.hop_length = config['hop_length']

        self.onset_threshold = config['onset_threshold']
        self.offset_threshold = config['offset_threshold']

        self.pitch_sum = config['pitch_sum']

        self.activation = nn.Sigmoid()

    def decode(self, pred, audio=None, f0=None):
        predictions = self._separate_raw_pred(pred)

        for k, v in predictions.items():
            predictions[k] = self.activation(v).squeeze()  # (T)

        onsets = _peak_selector(predictions['onset'], self.onset_threshold).to('cpu')
        offsets = _peak_selector(predictions['offset'], self.offset_threshold).to('cpu')
        frames = predictions['activation']

        if f0 is None:
            if audio is None:
                raise ValueError('Either audio or f0 should not be None.')
            else:
                f0, _, _ = librosa.pyin(audio.cpu().numpy(), fmin=65, fmax=2093, sr=self.sr,
                                        frame_length=self.win_length, hop_length=self.hop_length,
                                        fill_na=np.nan, center=True)
                f0 = torch.from_numpy(f0).float()

        pitches, intervals = _decode_notes(onsets, f0, self.pitch_sum, offsets=offsets, frames=frames)

        return pitches, intervals

    def _separate_raw_pred(self, pred):
        predictions = dict()

        predictions['onset'] = pred[:, :, 0]
        predictions['offset'] = pred[:, :, 1]
        predictions['activation'] = pred[:, :, 2]

        return predictions


def _peak_selector(pred, threshold):
    pred_peak = torch.zeros_like(pred)
    local_max_idx = 0
    for i in range(pred.size(0)):
        if pred[i] > threshold:
            if pred[i] > pred[local_max_idx]:
                local_max_idx = i
        else:
            if local_max_idx != 0:
                pred_peak[local_max_idx] = pred[local_max_idx]
                local_max_idx = 0

    return pred_peak


def _decode_notes(onsets, f0, pitch_sum, offsets=None, frames=None):
    pitches = []
    intervals = []

    f0_midi = torch.from_numpy(librosa.hz_to_midi(f0).squeeze())

    onset_diff = (torch.cat([onsets[:1], onsets[1:] - onsets[:-1]], dim=0) > 0).float()
    onset_diff_nonzero = onset_diff.nonzero()
    if offsets is not None:
        offset_diff = (torch.cat([offsets[:1], offsets[1:] - offsets[:-1]], dim=0) > 0).float()
    if frames is not None:
        frames_quantized = (frames >= 0.5).float().to('cpu')
        frame_diff = (torch.cat([frames_quantized[:-1] - frames_quantized[1:], frames_quantized[-1:]], dim=0) == 1).float()

    for i, nonzero in enumerate(onset_diff_nonzero):
        if i + 1 < onset_diff_nonzero.size(0):
            onset = nonzero[0].item()
            next_onset = onset_diff_nonzero[i + 1].item()
        else:  # last onset
            onset = nonzero[0].item()
            next_onset = onset_diff.size(0) - 1

        offset = None
        offset_confidence = 0
        frame_confidence = 0
        for i in range(onset + 2, next_onset):
            if offsets is not None:
                if offset_diff[i] == 1:
                    if offset_confidence < offsets[i]:
                        offset_confidence = offsets[i]
                        offset = i

            if frames is not None:
                if frame_diff[i] == 1:
                    new_confidence = 0
                    j = i + 1
                    while frames[j] < 0.5 and j < next_onset:
                        new_confidence = max(1 - frames[j], new_confidence)
                        j += 1
                    if frame_confidence < new_confidence:
                        frame_confidence = new_confidence
                        offset = i

        if offset is None:
            offset = next_onset - 1

        pitch_frames = f0_midi[onset: offset + 1]

        if pitch_sum == 'median':
            pitch = pitch_frames[~pitch_frames.isnan()].median().item()
        elif pitch_sum == 'weighted_mean':
            weighted_window = torch.hann_window(pitch_frames.size(0))
            weighted_pitch_frames = pitch_frames * weighted_window
            pitch = (weighted_pitch_frames[~weighted_pitch_frames.isnan()].sum() /
                        weighted_window[~weighted_pitch_frames.isnan()].sum()).item()
        elif pitch_sum == 'weighted_median':
            weighted_window = torch.hann_window(pitch_frames.size(0))
            weighted_window[pitch_frames.isnan()] = 0
            weighted_window /= weighted_window.sum()
            pitch = wquantiles.median(pitch_frames.cpu().numpy(), weighted_window.cpu().numpy())

        if pitch != pitch:  # check whether pitch is nan or not.
            pitch = 0

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset + 1])

    return pitches, intervals
