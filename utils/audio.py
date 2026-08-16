import torchaudio.transforms as T
import torch
import torchaudio
from utils.processors import AudioToMelProcessor
from config.identity_config import TrainingConfig

class AudioProcessor(torch.nn.Module):
    def __init__(self, sr=24000, n_fft=1024, hop_length=256, n_mels=100):
        # self.vocos_processor = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        super().__init__()
        self.manual_extractor = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=1,
        )

    def get_mel_spectrogram(self, waveform):
        # return self.vocos_processor.feature_extractor(waveform)    
        mel_spec = torch.log(torch.clip(self.manual_extractor(waveform), min=1e-7))
        return mel_spec

class AudioProcessorWrapper(AudioToMelProcessor):
    def __init__(self, config: TrainingConfig):
        self.ap = AudioProcessor(    
            sr=config.vocoder_sr,
            n_fft=config.vocoder_n_fft,
            hop_length=config.vocoder_hop_length,
            n_mels=config.vocoder_input_dim,
        )

        self.target_sr = config.vocoder_sr
        self.resemplers = {}
        
    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.target_sr:
            if sr not in self.resemplers:
                self.resemplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = self.resemplers[sr](waveform)
        target_mel = self.ap.get_mel_spectrogram(waveform)        
        return target_mel
