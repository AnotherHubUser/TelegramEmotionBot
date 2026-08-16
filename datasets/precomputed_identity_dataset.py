from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config.identity_config import TrainingConfig
from utils.masking import create_binary_mask


class PrecomputedIdentityDataset(Dataset):
    def __init__(self, config: TrainingConfig, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.mel_dir = self.cache_dir / config.mel_rel_dir
        self.feats_dir = self.cache_dir / config.feats_rel_dir
        self.feature_files = list(self.feats_dir.glob("**/*.pt"))

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feats_fp = self.feature_files[idx]

        relative_path = feats_fp.relative_to(self.feats_dir)
        mel_fp = self.mel_dir / relative_path
        
        feats_outputs = torch.load(feats_fp, map_location="cpu") # [13, T_feats, 768]
        target_mel = torch.load(mel_fp, map_location="cpu")         # [100, T_mel]
        
        return {
            self.config.batch_feats_key: feats_outputs,
            self.config.batch_mel_key: target_mel,
        }

### COLLATE_FN
def make_identity_cached_collate_fn(config: TrainingConfig):

    def cached_collate_fn(batch):
        # [100, T_mel] -> # [T_mel, 100] -> # [B, T_mel_max, 100]
        target_mels = pad_sequence([item[config.batch_mel_key].transpose(0, 1) for item in batch], batch_first=True)
        # [B, T_mel_max, 100] -> [B, 100, T_mel_max]
        target_mels = target_mels.transpose(1, 2)
        target_mels_lens = torch.tensor([item[config.batch_mel_key].shape[-1] for item in batch], dtype=torch.long)
        max_mel_len = target_mels.shape[-1]
        target_mels_mask = create_binary_mask(max_mel_len, target_mels_lens)
        
        # [13, T_feats, 768] -> # [T_feats, 13, 768] -> # [B, T_feat_max, 13, 768]
        feats_outputs = pad_sequence([item[config.batch_feats_key].permute(1, 0, 2) for item in batch], batch_first=True)
        # [B, T_feat_max, 13, 768] -> [B, 13, T_feat_max, 768]
        feats_outputs = feats_outputs.permute(0, 2, 1, 3)
        feats_outputs_lens = torch.tensor([item[config.batch_feats_key].shape[1] for item in batch], dtype=torch.long)
        max_feats_outputs_len = feats_outputs.shape[2]
        feats_outputs_mask = create_binary_mask(max_feats_outputs_len, feats_outputs_lens)
             
        return {
            config.batch_feats_key: feats_outputs,
            config.batch_feats_mask_key: feats_outputs_mask,
            
            config.batch_mel_key: target_mels,
            config.batch_mel_mask_key: target_mels_mask,
        }
    
    return cached_collate_fn