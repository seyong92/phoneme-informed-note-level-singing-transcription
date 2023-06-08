import torch
from torch import nn
from torchaudio import transforms as T
from nnAudio import features as S
from .feature import FeatureExtractor
from .subnetworks import ConvStack, BiLSTM


class PhonemeRecognitionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.input_features = config['n_mels']
        self.output_features = 39

        self.conv_stack, self.rnn, self.fc = self._create_model(self.input_features,
                                                                self.output_features, config)

        self.feat_ext = FeatureExtractor(config)

    def _create_model(self, input_features, output_features, config):
        model_complexity = config['model_complexity']
        model_size = model_complexity * 16

        conv_stack = ConvStack(input_features, model_size)
        rnn = BiLSTM(model_size, model_size // 2, num_layers=1)
        fc = nn.Linear(model_size, output_features)

        return conv_stack, rnn, fc

    def forward(self, data):
        x = self.conv_stack(data)
        x = self.rnn(x)
        x = self.fc(x)
        return x

    def run_on_batch(self, batch):
        feat = self.feat_ext(batch['audio']).transpose(1, 2).unsqueeze(1)  # (N, 1, T, F)
        pred = self(feat)  # (N, T, F)

        predictions = {
            'frame': pred,
        }

        return predictions
