import torch
import torch.nn as nn
import time
from typing import Dict
from config.identity_config import TrainingConfig
from trainers.base_trainer import BaseTrainer


class IdentityTrainer(BaseTrainer):
    def __init__(self, config: TrainingConfig, models: Dict[str, nn.Module], optimizer: torch.optim.Optimizer, criterion=None):
        super().__init__(config, models, optimizer)
        self.criterion = criterion or nn.L1Loss(reduction='none')
        self.pooling = self.models[config.models_pooling_name]
        self.adapter = self.models[config.models_adapter_name]
                
    def train_step(self, batch_dict):
        # start_step_time = time.perf_counter()
        self.pooling.train()
        self.adapter.train()
        self.optimizer.zero_grad()
        # print(f"set train mode: {time.perf_counter() - start_step_time:.4f} secs.")
    
        feats_outputs = batch_dict[self.config.batch_feats_key].to(self.device)
        feats_outputs_mask = batch_dict[self.config.batch_feats_mask_key].to(self.device)

        target_mel = batch_dict[self.config.batch_mel_key].to(self.device)
        mel_mask = batch_dict[self.config.batch_mel_mask_key].to(self.device)
        target_len = target_mel.shape[-1]

        # start_step_time = time.perf_counter()
        pooling_feats = self.pooling(feats_outputs) * feats_outputs_mask.unsqueeze(2)
        # print(f"passing pooling: {time.perf_counter() - start_step_time:.4f} secs.")
        
        # start_step_time = time.perf_counter()
        predicted_mel = self.adapter(pooling_feats, target_len, mask=feats_outputs_mask, mel_mask=mel_mask)
        # print(f"passing adapter: {time.perf_counter() - start_step_time:.4f} secs.")
        
        # start_step_time = time.perf_counter()
        mel_mask = mel_mask.unsqueeze(1)
        raw_loss = self.criterion(predicted_mel, target_mel) * mel_mask
        # expanded_mask = mel_mask.unsqueeze(1)
        # loss = (raw_loss * expanded_mask).sum() / (expanded_mask.sum() * predicted_mel.shape[1] + 1e-7)
        loss = raw_loss.sum() / (mel_mask.sum() * self.config.vocoder_input_dim)
        # print(f"computing loss: {time.perf_counter() - start_step_time:.4f} secs.")
        
        # start_step_time = time.perf_counter()
        loss.backward()
        # print(f"loss backward: {time.perf_counter() - start_step_time:.4f} secs.")
        
        # start_step_time = time.perf_counter()
        first_layer_grad = self.adapter.input_proj.weight.grad
        grad_norm = first_layer_grad.norm().item() if first_layer_grad is not None else 0.0
        
        total_grad_norm = nn.utils.clip_grad_norm_(self.adapter.parameters(), self.config.max_grad_norm)
        nn.utils.clip_grad_norm_(self.pooling.parameters(), self.config.max_grad_norm)
        # print(f"clipping grads and etc: {time.perf_counter() - start_step_time:.4f} secs.")
        
        # start_step_time = time.perf_counter()
        self.optimizer.step()
        # print(f"optimizer step: {time.perf_counter() - start_step_time:.4f} secs.")
        
        return {
            self.config.step_loss_key: loss.item(),
            self.config.step_grad_norm_key: grad_norm,
            self.config.step_total_grad_norm_key: total_grad_norm,
            self.config.step_predicted_mel_key: predicted_mel.detach(),
            self.config.step_target_mel_key: target_mel.detach(),
            self.config.step_mel_mask_key: mel_mask.detach()
        }
    
    @torch.no_grad()
    def evaluate_sanity(self, batch_dict):
        self.adapter.eval()
        self.pooling.eval()
        
        feats_outputs = batch_dict[self.config.batch_feats_key].to(self.device)
        feats_outputs_mask = batch_dict[self.config.batch_feats_mask_key].to(self.device)

        target_mel = batch_dict[self.config.batch_mel_key].to(self.device)
        mel_mask = batch_dict[self.config.batch_mel_mask_key].to(self.device)
        target_len = target_mel.shape[-1]
        
        pooling_feats = self.pooling(feats_outputs) * feats_outputs_mask.unsqueeze(2)
        predict_real = self.adapter(pooling_feats, target_len, mask=feats_outputs_mask, mel_mask=mel_mask)
        
        loss_real_raw = self.criterion(predict_real, target_mel)
        # expanded_mask = mel_mask.unsqueeze(1)
        # loss_real = (loss_real_raw * expanded_mask).sum() / (expanded_mask.sum() * predict_real.shape[1] + 1e-7)
        mel_mask = mel_mask.unsqueeze(1)
        loss_real = (loss_real_raw * mel_mask).sum() / (mel_mask.sum() * self.config.vocoder_input_dim)
        
        noise_feats = torch.randn_like(pooling_feats) * feats_outputs_mask.unsqueeze(2)
        predict_noise = self.adapter(noise_feats, target_len, mask=feats_outputs_mask, mel_mask=mel_mask)
        
        loss_noise_raw = self.criterion(predict_noise, target_mel)
        # loss_noise = (loss_noise_raw * expanded_mask).sum() / (expanded_mask.sum() * predict_noise.shape[1] + 1e-7)
        loss_noise = (loss_noise_raw * mel_mask).sum() / (mel_mask.sum() * self.config.vocoder_input_dim)
        
        return {
            self.config.step_loss_key: loss_real.item(),
            self.config.step_loss_noise_key: loss_noise.item(),
            self.config.step_predicted_mel_key: predict_real,
            self.config.step_target_mel_key: target_mel,
        }


# class IdentityTrainer(BaseTrainer):
#     def __init__(self, config: TrainingConfig, model=None, pooling_module=None, hubert=None, audio_processor=None, vocoder=None, mask_manager=None, optimizer=None):
#         super().__init__(config, model, pooling_module, hubert, optimizer)
        
#         self.ap = (audio_processor or AudioProcessor(
#             sr=self.config.vocoder_sr,
#             n_fft=self.config.vocoder_n_fft,
#             hop_length=self.config.vocoder_hop_length,
#             n_mels=self.config.vocoder_input_dim,
#         )).to(self.device)

#         self.mask_manager = (mask_manager or MaskManager(
#             orig_sr=self.config.orig_sr,
#             hubert_sr=self.config.hubert_sr,
#             vocos_sr=self.config.vocoder_sr,
#             hop_length=self.config.vocoder_hop_length,
#         ))

#         self.vocoder = (vocoder or Vocoder(model_name=self.config.vocoder_model_name)).to(self.device)

#         self.criterion = nn.L1Loss(reduction='none')

#     def _extract_hubert(self, batch_dict):
#         waveform = batch_dict["anchor"].to(self.device)
#         orig_lens = batch_dict["anchor_lens"].to(self.device)
        
#         wave_hubert = self.resampler_orig_to_hubert(waveform)
#         hubert_mask = self.mask_manager.get_hubert_wav_mask(orig_lens, wave_hubert.shape[-1]).to(self.device)
        
#         return wave_hubert, hubert_mask
    
#     def _extract_mel(self, batch_dict):
#         waveform = batch_dict["anchor"].to(self.device)
#         orig_lens = batch_dict["anchor_lens"].to(self.device)
        
#         wave_vocoder = self.resampler_orig_to_vocoder(waveform)
#         target_mel = self.ap.get_mel_spectrogram(wave_vocoder)
        
#         mel_mask = self.mask_manager.get_mel_mask(orig_lens, target_mel.shape[-1]).to(self.device)
#         return target_mel, mel_mask

#     def _prepare_batch(self, batch_dict):
#         waveform = batch_dict["anchor"].to(self.device)
#         orig_lens = batch_dict["anchor_lens"].to(self.device)
        
#         wave_hubert = self.resampler_orig_to_hubert(waveform)
#         wave_vocoder = self.resampler_orig_to_vocoder(waveform)
        
#         target_mel = self.ap.get_mel_spectrogram(wave_vocoder)
        
#         hubert_mask = self.mask_manager.get_hubert_wav_mask(orig_lens, wave_hubert.shape[-1]).to(self.device)
#         mel_mask = self.mask_manager.get_mel_mask(orig_lens, target_mel.shape[-1]).to(self.device)
        
#         return wave_hubert, hubert_mask, target_mel, mel_mask

#     def _check_hubert_step(self, batch_dict):
#         wave_hubert, _, _, _ = self._prepare_batch(batch_dict)
        
#         start_step_time = time.perf_counter()
#         self.hubert.hubert(wave_hubert)
#         print(f"PASSING BARE HUBERT: {time.perf_counter() - start_step_time:.4f} secs.")
        
#     def get_hubert_feats(self, batch_dict):
#         # start_step_time = time.perf_counter()
#         wave_hubert, hubert_mask = self._extract_hubert(batch_dict)
#         # print(f"prepared batch: {time.perf_counter() - start_step_time:.4f} secs.")
    
#         start_step_time = time.perf_counter()
#         hubert_outputs, hubert_mask_out = self.hubert(wave_hubert, mask=hubert_mask)
#         print(f"PASSING HUBERT: {time.perf_counter() - start_step_time:.4f} secs.")
#         hubert_outputs = torch.stack(hubert_outputs.hidden_states)
#         print(hubert_outputs.shape)
#         print(hubert_mask_out.shape)
#         return hubert_outputs, hubert_mask_out
        
#     def train_step(self, batch_dict, hubert_outputs, hubert_mask_out):
#         # start_step_time = time.perf_counter()
#         self.model.train()
#         self.pooling.train()
#         self.optimizer.zero_grad()
#         # print(f"set train mode: {time.perf_counter() - start_step_time:.4f} secs.")
    
#         target_mel, mel_mask = self._extract_mel(batch_dict=batch_dict)
#         target_len = target_mel.shape[-1]

#         # start_step_time = time.perf_counter()
#         pooling_feats = self.pooling(hubert_outputs) * hubert_mask_out.unsqueeze(2)
#         # print(f"passing pooling: {time.perf_counter() - start_step_time:.4f} secs.")
        
#         # start_step_time = time.perf_counter()
#         predicted_mel = self.model(pooling_feats, target_len, mask=hubert_mask_out, mel_mask=mel_mask)
#         # print(f"passing adapter: {time.perf_counter() - start_step_time:.4f} secs.")
        
#         # start_step_time = time.perf_counter()
#         mel_mask = mel_mask.unsqueeze(1)
#         raw_loss = self.criterion(predicted_mel, target_mel) * mel_mask
#         # expanded_mask = mel_mask.unsqueeze(1)
#         # loss = (raw_loss * expanded_mask).sum() / (expanded_mask.sum() * predicted_mel.shape[1] + 1e-7)
#         loss = raw_loss.sum() / (mel_mask.sum() * self.config.vocoder_input_dim)
#         # print(f"computing loss: {time.perf_counter() - start_step_time:.4f} secs.")
        
#         # start_step_time = time.perf_counter()
#         loss.backward()
#         # print(f"loss backward: {time.perf_counter() - start_step_time:.4f} secs.")
        
#         # start_step_time = time.perf_counter()
#         first_layer_grad = self.model.input_proj.weight.grad
#         grad_norm = first_layer_grad.norm().item() if first_layer_grad is not None else 0.0
        
#         total_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
#         nn.utils.clip_grad_norm_(self.pooling.parameters(), self.config.max_grad_norm)
#         # print(f"clipping grads and etc: {time.perf_counter() - start_step_time:.4f} secs.")
        
#         # start_step_time = time.perf_counter()
#         self.optimizer.step()
#         # print(f"optimizer step: {time.perf_counter() - start_step_time:.4f} secs.")
        
#         return {
#             "loss": loss.item(),
#             "grad_norm": grad_norm,
#             "total_grad_norm": total_grad_norm,
#             "predicted_mel": predicted_mel.detach(),
#             "target_mel": target_mel.detach(),
#             "mel_mask": mel_mask.detach()
#         }
    
#     def train_full_step(self, batch_dict):
#         hubert_outputs, hubert_mask_out = self.get_hubert_feats(batch_dict)
#         return self.train_step(batch_dict, hubert_outputs, hubert_mask_out)


#     @torch.no_grad()
#     def evaluate_sanity(self, batch_dict):
#         self.model.eval()
#         self.pooling.eval()
        
#         wave_hubert, hubert_mask, target_mel, mel_mask = self._prepare_batch(batch_dict)
#         target_len = target_mel.shape[-1]
        
#         hubert_outputs, hubert_mask_out = self.hubert(wave_hubert, mask=hubert_mask)
#         hubert_feats = self.pooling(hubert_outputs) * hubert_mask_out.unsqueeze(2)
#         predict_real = self.model(hubert_feats, target_len, mask=hubert_mask_out, mel_mask=mel_mask)
        
#         loss_real_raw = self.criterion(predict_real, target_mel)
#         # expanded_mask = mel_mask.unsqueeze(1)
#         # loss_real = (loss_real_raw * expanded_mask).sum() / (expanded_mask.sum() * predict_real.shape[1] + 1e-7)
#         mel_mask = mel_mask.unsqueeze(1)
#         loss_real = (loss_real_raw * mel_mask).sum() / (mel_mask.sum() * self.config.vocoder_input_dim)
        
#         noise_feats = torch.randn_like(hubert_feats) * hubert_mask_out.unsqueeze(2)
#         predict_noise = self.model(noise_feats, target_len, mask=hubert_mask_out, mel_mask=mel_mask)
        
#         loss_noise_raw = self.criterion(predict_noise, target_mel)
#         # loss_noise = (loss_noise_raw * expanded_mask).sum() / (expanded_mask.sum() * predict_noise.shape[1] + 1e-7)
#         loss_noise = (loss_noise_raw * mel_mask).sum() / (mel_mask.sum() * self.config.vocoder_input_dim)
        
#         return {
#             "loss_real": loss_real.item(),
#             "loss_noise": loss_noise.item(),
#             "predict_real": predict_real,
#             "target_mel": target_mel
#         }
