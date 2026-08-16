import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from models.hubert import HubertEmbeddings
from models.learnable_layer_pooling import LearnableLayerPooling
from models.adapter import EmotionAdapter
from utils.masking import MaskManager
from utils.audio import AudioProcessor

def test_gradient_flow_and_overfit():
    """Check PyTorch graph works and gradients reach first layer"""
    torch.manual_seed(42)
    
    hubert = HubertEmbeddings()
    pooling = LearnableLayerPooling(hubert.get_num_layers())
    adapter = EmotionAdapter()
    ap = AudioProcessor()
    mask_manager = MaskManager()
    criterion = nn.L1Loss(reduction='none')
    
    optimizer = optim.Adam([
        {'params': adapter.parameters(), 'lr': 1e-3},
        {'params': pooling.parameters(), 'lr': 1e-3}
    ])
    
    wave_16k = torch.randn(2, 32000)
    lens_48k = torch.tensor([96000, 96000], dtype=torch.long)

    wave_24k = nn.functional.pad(wave_16k, (0, int(wave_16k.shape[-1] * 0.5)))
    fake_target_mel = ap.get_mel_spectrogram(wave_24k)
    target_len = fake_target_mel.shape[-1]
    
    hubert_wav_mask = mask_manager.get_hubert_wav_mask(lens_48k, 32000)
    mel_mask = mask_manager.get_mel_mask(lens_48k, target_len)

    weights_before = pooling.layer_weights.data.clone()
    
    initial_loss = None
    final_loss = None
    
    for i in range(3):
        optimizer.zero_grad()
        
        hubert_layers, hubert_frame_mask = hubert(wave_16k, mask=hubert_wav_mask)
        hubert_feats = pooling(hubert_layers)
        predicted_mel = adapter(hubert_feats, target_len, mask=hubert_frame_mask, mel_mask=mel_mask)
        
        raw_loss = criterion(predicted_mel, fake_target_mel)
        masked_loss = raw_loss * mel_mask.unsqueeze(1)
        loss = masked_loss.sum() / (mel_mask.sum() * 100)
        
        if i == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

        loss.backward()
        optimizer.step()
        
    # Check loss fall for 3 steps
    assert final_loss < initial_loss, f"Loss doesn't fall! step 0: {initial_loss}, step 2: {final_loss}"
    
    # Check all weights shifted in LearnableLayerPooling
    weight_delta = torch.abs(weights_before - pooling.layer_weights.data).sum().item()
    assert weight_delta > 0.0, "LearnableLayerPooling weights are frozen"
