import torch
import torchaudio
import time
from typing import List
from transformers import HubertModel
from utils.processors import AudioToEmbeddingsProcessor
from config.identity_config import TrainingConfig
from torch.nn.utils.rnn import pad_sequence
from utils.masking import create_binary_mask

class HubertEmbeddings(torch.nn.Module):
    def __init__(self, model_name="facebook/hubert-base-ls960"):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(model_name)
        self.hubert.eval()
        for param in self.hubert.parameters():
            param.requires_grad = False

        self.hubert.config.num_hidden_layers

    def get_num_layers(self):
        return self.hubert.config.num_hidden_layers + 1
    
    @torch.no_grad()
    def fast_forward(self, waveform):
        return self.hubert(waveform, output_hidden_states=True)

    @torch.no_grad()
    def forward(self, waveform, mask=None):
        """
        waveform: [B, L_16k]
        """
        if mask is not None:
            # start_step_time = time.perf_counter()
            masked_waveform = waveform * mask
            
            counts = mask.sum(dim=-1, keepdim=True) # [B, 1]
            mean = masked_waveform.sum(dim=-1, keepdim=True) / counts # [B, 1]
            
            sq_diff = ((masked_waveform - mean) ** 2) * mask
            var = sq_diff.sum(dim=-1, keepdim=True) / counts # [B, 1]
            
            input_values = (masked_waveform - mean) / torch.sqrt(var + 1e-7)
            input_values = input_values * mask
            # print(f"preparing inputs: {time.perf_counter() - start_step_time:.4f} secs.")
            


            # ----- ВОЗВРАЩАЕМ БАТЧИНГ -------
            extract_features = self.hubert.feature_extractor(input_values).transpose(1, 2)
            attention_mask = self.hubert._get_feature_vector_attention_mask(extract_features.shape[1], mask)

            position_embeddings = self.hubert.feature_projection(extract_features) * attention_mask.unsqueeze(2)
              
            encoder_outputs = self.hubert.encoder(
                position_embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            return encoder_outputs, attention_mask
                        

            # # --- ЭТОЙ ЧАСТИ НЕ БЫЛО ---
            # # start_step_time = time.perf_counter()
            # live_lengths = mask.sum(dim=-1).int()

            # features_list = []
            # B = input_values.shape[0]
            # # print(f"preparing before for: {time.perf_counter() - start_step_time:.4f} secs.")
            
            # for i in range(B):
            #     # start_step_time = time.perf_counter()
            #     live_len = live_lengths[i]
            #     single_wave = input_values[i:i+1, :live_len] 
            #     single_feats = self.hubert.feature_extractor(single_wave) # [1, 512, T_live]
            #     features_list.append(single_feats)
            #     # print(f"{i} in for extractor: {time.perf_counter() - start_step_time:.4f} secs.")
            

            # # max_feat_len = max(f.shape[-1] for f in features_list)
            # # outputs_like = self.hubert(input_values, attention_mask=mask, output_hidden_states=True)
            # # max_feat_len = outputs_like.last_hidden_state.shape[1]
            # # kostul', no tak
            # # start_step_time = time.perf_counter()
            # max_feat_len = (input_values.shape[-1] - 80) // 320

            # padded_features = [torch.nn.functional.pad(f, (0, max_feat_len - f.shape[-1])) for f in features_list]     
            # extract_features = torch.cat(padded_features, dim=0) # [B, 512, max_feat_len]
            # # return extract_features, mask

            # # extract_features = torch.zeros((B, 512, max_feat_len), dtype=features_list[0].dtype, device=features_list[0].device)
            # # for i, f in enumerate(features_list):
            # #     extract_features[i, :, :f.shape[-1]] = f[0] # f is [1, 512, T_live]
            # extract_features = extract_features.transpose(1, 2) # [B, max_feat_len, 512]
            # # print(f"preparing extract features: {time.perf_counter() - start_step_time:.4f} secs.")
            
            # # start_step_time = time.perf_counter()
            # attention_mask = self.hubert._get_feature_vector_attention_mask(extract_features.shape[1], mask)
            # # print(f"get attention mask: {time.perf_counter() - start_step_time:.4f} secs.")
            # # return extract_features.transpose(1, 2), attention_mask

            # # start_step_time = time.perf_counter()
            # position_embeddings = self.hubert.feature_projection(extract_features) * attention_mask.unsqueeze(2)
            # # print(f"projection: {time.perf_counter() - start_step_time:.4f} secs.")
            # # return position_embeddings.transpose(1, 2), attention_mask
            
            # # start_step_time = time.perf_counter()
            # encoder_outputs = self.hubert.encoder(
            #     position_embeddings,
            #     attention_mask=attention_mask,
            #     output_hidden_states=True,
            #     return_dict=True
            # )
            # # print(f"passing encoder: {time.perf_counter() - start_step_time:.4f} secs.")
            # # print("out hubert")

            # return encoder_outputs, attention_mask
            # # --- ЭТОЙ ЧАСТИ НЕ БЫЛО ---
            

        else:
            mean = waveform.mean(dim=-1, keepdim=True)
            var = waveform.var(dim=-1, keepdim=True)
            input_values = (waveform - mean) / torch.sqrt(var + 1e-7)

        # print(input_values.shape)
        outputs = self.hubert(input_values, attention_mask=mask, output_hidden_states=True)
        
        # if mask is not None:
        #     mask = self.hubert._get_feature_vector_attention_mask(outputs.last_hidden_state.shape[1], mask)
        
        return outputs, mask

class HubertWrapper(AudioToEmbeddingsProcessor):
    def __init__(self, config: TrainingConfig):
        self.device = config.device
        self.model = HubertEmbeddings(model_name=config.feats_model_name).to(self.device)
        self.target_sr = config.feats_sr
        self.device = config.device
        self.resemplers = torch.nn.ModuleDict()
        
    def __call__(self, waveform: torch.Tensor | List[torch.Tensor], sr: int | List[int]) -> torch.Tensor:
        if isinstance(sr, int):
            waveform = waveform.to(self.device)
            if sr != self.target_sr:
                sr_str = str(sr)
                if sr_str not in self.resemplers:
                    self.resemplers[sr_str] = torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)
                waveform = self.resemplers[sr_str](waveform)
            
            hubert_outputs = self.model.fast_forward(waveform)
            hubert_outputs = torch.stack(hubert_outputs.hidden_states)
            return hubert_outputs

        # Multiple processing
        waveforms = waveform
        srs = sr
        resampled_waveforms = []
        for waveform, sr in zip(waveforms, srs):
            waveform = waveform.to(self.device)
            if sr != self.target_sr:
                sr_str = str(sr)
                if sr_str not in self.resemplers:
                    self.resemplers[sr_str] = torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)
                waveform = self.resemplers[sr_str](waveform)
            resampled_waveforms.append(waveform.squeeze())
            
        padded_waveforms = pad_sequence(resampled_waveforms, batch_first=True).to(self.device)
        
        lengths = torch.tensor([w.shape[-1] for w in resampled_waveforms], device=self.device)
        max_padded_wave_len = padded_waveforms.shape[-1]
        mask = create_binary_mask(max_padded_wave_len, lengths)

        feats, out_mask = self.model(padded_waveforms, mask)
        return torch.stack(feats.hidden_states, dim=1), out_mask
    