import pytest
import torch
from models.hubert import HubertEmbeddings
from models.learnable_layer_pooling import LearnableLayerPooling
from models.adapter import EmotionAdapter
from utils.masking import MaskManager

def test_pipeline_shapes():
    """Проверяем, что оси тензоров сходятся на каждом этапе пайплайна"""
    B = 2
    L_max_48k = 96000  # 2 secs
    
    hubert = HubertEmbeddings()
    pooling = LearnableLayerPooling(hubert.get_num_layers())
    adapter = EmotionAdapter()
    mask_manager = MaskManager()
    
    fake_lens_48k = torch.tensor([48000, 96000], dtype=torch.long)  # 1 and 2 secs
    L_max_16k = L_max_48k // 3
    fake_wave_16k = torch.randn(B, L_max_16k)
 
    hubert_wav_mask = mask_manager.get_hubert_wav_mask(fake_lens_48k, L_max_16k)
    assert hubert_wav_mask.shape == (B, L_max_16k)
    
    # Check HuBERT
    hubert_layers, hubert_frame_mask = hubert(fake_wave_16k, mask=hubert_wav_mask)
    assert len(hubert_layers.hidden_states) == 13
    assert hubert_layers.last_hidden_state.shape[0] == B
    
    # Check LearnableLayerPooling
    hubert_feats = pooling(hubert_layers)
    assert hubert_feats.shape == (B, hubert_layers[0].shape[1], 768)
    
    # Check Adapter
    target_len = 375
    mel_mask = mask_manager.get_mel_mask(fake_lens_48k, target_len)
    
    predicted_mel = adapter(
        hubert_feats, 
        target_len=target_len, 
        mask=hubert_frame_mask, 
        mel_mask=mel_mask
    )
    
    # Check [B, 100, T_mel_max]
    assert predicted_mel.shape == (B, 100, target_len)
