import os
import torch
import torch.nn as nn
from typing import Dict
from config.identity_config import TrainingConfig


class BaseTrainer:
    def __init__(self, config: TrainingConfig, models: Dict[str, nn.Module], optimizer: torch.optim.Optimizer):
        self.config = config
        self.device = config.device
        self.scaler = None
        if 'cuda' in str(self.device):
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.models = nn.ModuleDict(models).to(self.device)
        self.optimizer = optimizer

    def save_checkpoint(self, epoch: int, loss: float, filename: str):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
        models_state = {name: model.state_dict() for name, model in self.models.items()}

        state_snapshot = {
            self.config.snapshot_epoch_key: epoch,
            self.config.snapshot_loss_key: loss,
            self.config.snapshot_models_state_key: models_state,
            self.config.snapshot_optimizer_state_key: self.optimizer.state_dict(),
        }
        torch.save(state_snapshot, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        snapshot = torch.load(checkpoint_path, map_location=self.device)
        
        for name, model in self.models.items():
            if name in snapshot[self.config.snapshot_models_state_key]:
                model.load_state_dict(snapshot[self.config.snapshot_models_state_key][name])
                
        self.optimizer.load_state_dict(snapshot[self.config.snapshot_optimizer_state_key])
        return snapshot[self.config.snapshot_epoch_key], snapshot[self.config.snapshot_loss_key]

# class BaseTrainer:
#     def __init__(self, config: TrainingConfig, model=None, pooling_module=None, hubert=None, optimizer=None):
#         self.config = config
#         self.device = config.device
        
#         self.hubert = (hubert or HubertEmbeddings(model_name=self.config.hubert_model_name)).to(self.device)
#         self.model = (model or EmotionAdapter(
#             input_dim=self.config.hubert_output_dim, 
#             hidden_dim=self.config.adapter_hidden_dim, 
#             output_dim=self.config.vocoder_input_dim)).to(self.device)
#         self.pooling = (pooling_module or LearnableLayerPooling(num_layers=self.hubert.get_num_layers())).to(self.device)
#         self.optimizer = optimizer or optim.Adam([
#             {'params': self.model.parameters(), 'lr': self.config.adapter_lr},
#             {'params': self.pooling.parameters(), 'lr': self.config.pooling_lr}
#         ])

#         self.resampler_orig_to_vocoder = torchaudio.transforms.Resample(config.orig_sr, config.vocoder_sr).to(self.device)
#         self.resampler_orig_to_hubert = torchaudio.transforms.Resample(config.orig_sr, config.hubert_sr).to(self.device)

#     def save_checkpoint(self, epoch, loss, filename):
#         os.makedirs(self.config.checkpoint_dir, exist_ok=True)
#         checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        
#         state_snapshot = {
#             "epoch": epoch,
#             "loss": loss,
#             "model_state_dict": self.model.state_dict(),
#             "pooling_state_dict": self.pooling.state_dict(),
#             "optimizer_state_dict": self.optimizer.state_dict(),
#         }
        
#         torch.save(state_snapshot, checkpoint_path)
        
#     def load_checkpoint(self, checkpoint_path):
#         snapshot = torch.load(checkpoint_path, map_location=self.device)
        
#         self.model.load_state_dict(snapshot["model_state_dict"])
#         self.pooling.load_state_dict(snapshot["pooling_state_dict"])
#         self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        
#         return snapshot["epoch"], snapshot["loss"]
