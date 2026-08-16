from pathlib import Path
import torchaudio
import torch
from torch.utils.data import Dataset
from config.identity_config import TrainingConfig


class LibriTTSRDataset(Dataset):
    def __init__(self, config: TrainingConfig, dataset_dir: str | Path, transform=None):
        self.config = config
        self.transform = transform
        self.dataset_dir = Path(dataset_dir)
        self.audio_files = list(self.dataset_dir.glob("**/*.wav"))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        fp = self.audio_files[idx]
        relative_path = fp.relative_to(self.dataset_dir)
        
        wave, sr = torchaudio.load(fp)
        if wave.shape[0] > 1:
            wave = torch.mean(wave, dim=0, keepdim=True)

        if self.transform:
            wave = self.transform(wave)
        
        # wave is [1, Time]
        return {
            self.config.batch_wave_key: wave,
            self.config.batch_sr_key: sr,
            self.config.batch_path_key: relative_path
        }
