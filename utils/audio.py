import torchaudio.transforms as T
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from typing import List
from utils.processors import AudioToMelProcessor
from utils.masking import create_binary_mask
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
        self.device = config.device
        
        self.ap = AudioProcessor(    
            sr=config.vocoder_sr,
            n_fft=config.vocoder_n_fft,
            hop_length=config.vocoder_hop_length,
            n_mels=config.vocoder_input_dim,
        ).to(self.device)

        self.target_sr = config.vocoder_sr
        self.hop_length = config.vocoder_hop_length
        self.resemplers = torch.nn.ModuleDict()
        
    def __call__(self, waveform: torch.Tensor | List[torch.Tensor], sr: int | List[int]) -> torch.Tensor:
        if isinstance(sr, int):
            if sr != self.target_sr:
                sr_str = str(sr)  # ModuleDict требует строковые ключи
                if sr_str not in self.resemplers:
                    self.resemplers[sr_str] = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = self.resemplers[sr](waveform)
            target_mel = self.ap.get_mel_spectrogram(waveform)        
            return target_mel
        
        # Multiple processing
        waveforms = waveform
        srs = sr
        resampled_waveforms = []
        for waveform, sr in zip(waveforms, srs):
            waveform = waveform.to(self.device)
            if sr != self.target_sr:
                sr_str = str(sr)
                if sr_str not in self.resemplers:
                    self.resemplers[sr_str] = T.Resample(sr, self.target_sr).to(self.device)
                waveform = self.resemplers[sr_str](waveform)
            resampled_waveforms.append(waveform.squeeze())
            
        lengths = torch.tensor([w.shape[-1] for w in resampled_waveforms], device=self.device)
        padded_waveforms = pad_sequence(resampled_waveforms, batch_first=True).to(self.device)

        target_mels = self.ap.get_mel_spectrogram(padded_waveforms)
        
        mel_lengths = (lengths // self.hop_length) + 1
        max_mel_len = target_mels.shape[-1]
        
        mel_masks = create_binary_mask(max_mel_len, mel_lengths)
        return target_mels, mel_masks
