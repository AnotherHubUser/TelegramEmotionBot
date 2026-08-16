import torch
from vocos import Vocos
from utils.processors import MelToAudioProcessor
from config.identity_config import TrainingConfig

class Vocoder(torch.nn.Module):
    def __init__(self, model_name="charactr/vocos-mel-24khz"):
        super().__init__()
        self.model = Vocos.from_pretrained(model_name)

    @torch.no_grad()
    def generate(self, spectrogram):
        """
        Принимает спектрограмму [Batch, Mel_channels, Time]
        Возвращает аудио [Batch, Samples]
        """
        return self.model.decode(spectrogram)

class VocoderWrapper(MelToAudioProcessor):
    def __init__(self, config: TrainingConfig):
        self.device = config.device
        self.vocoder = Vocoder(config.vocoder_model_name).to(config.device)
        
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.vocoder.generate(spectrogram.to(self.device))
