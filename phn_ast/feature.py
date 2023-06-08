from torch import nn
from torchaudio import transforms as T
from nnAudio import features as S


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat = S.MelSpectrogram(sr=config['sample_rate'],
                                        n_fft=config['win_length'],
                                        win_length=config['win_length'],
                                        n_mels=config['n_mels'],
                                        hop_length=config['hop_length'],
                                        fmin=config['fmin'],
                                        fmax=config['fmax'],
                                        center=True)
        self.db = T.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, audio):
        feature = self.feat(audio)
        feature = self.db(feature)

        return feature

    def _freeze(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self._freeze(child)
