import pytest
import torch
import torch.nn as nn
import torchaudio
from models.hubert import HubertEmbeddings
from models.learnable_layer_pooling import LearnableLayerPooling
from models.adapter import EmotionAdapter
from utils.masking import MaskManager
from utils.audio import AudioProcessor


def test_padding_invariance():
    """Тест братишки: лосс живого аудио не должен меняться от размера паддинга"""
    torch.manual_seed(42)
    
    hubert = HubertEmbeddings()
    pooling = LearnableLayerPooling(hubert.get_num_layers())
    adapter = EmotionAdapter()
    ap = AudioProcessor()
    mask_manager = MaskManager()
    criterion = nn.L1Loss(reduction='none')
    
    # Наше живое аудио (1 секунда звука)
    live_len_48k = 48000
    live_wave_48k = torch.randn(1, live_len_48k)

    resampler_48k_to_24k = torchaudio.transforms.Resample(48000, 24000)
    resampler_48k_to_16k = torchaudio.transforms.Resample(48000, 16000)

    # Scenario А: Min padding (up to 60000 samples)
    max_len_A = 60000
    wave_A_48k = torch.cat([live_wave_48k, torch.zeros(1, max_len_A - live_len_48k)], dim=-1)
    lens_48k_A = torch.tensor([live_len_48k])
    # print("wave_A_48k", wave_A_48k.shape)

    
    # Scenario Б: Large padding (up to 180000 samples)
    max_len_B = 180715
    wave_B_48k = torch.cat([live_wave_48k, torch.zeros(1, max_len_B - live_len_48k)], dim=-1)
    lens_48k_B = torch.tensor([live_len_48k])
    # print("wave_B_48k", wave_B_48k.shape)
    
    
    def assert_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor, name_a: str, name_b: str, pad_value=0.0) -> None:
        a_extended = nn.functional.pad(tensor_a, (0, int(tensor_b.shape[-1] - tensor_a.shape[-1])), value=pad_value)
        assert torch.allclose(tensor_b, a_extended, atol=1e-4), \
        f"{name_b}: {tensor_b}, {name_a}_extended: {a_extended}"

    assert_tensors(wave_A_48k, wave_B_48k, "wave_A_48k", "wave_B_48k")

    # Create 16k waves
    wave_A_16k = resampler_48k_to_16k(wave_A_48k)
    wave_B_16k = resampler_48k_to_16k(wave_B_48k)
    assert_tensors(wave_A_16k, wave_B_16k, "wave_A_16k", "wave_B_16k")
    
    # Create mask for hubert inputs
    hubert_A_mask = mask_manager.get_hubert_wav_mask(lens_48k_A, wave_A_16k.shape[-1])
    hubert_B_mask = mask_manager.get_hubert_wav_mask(lens_48k_B, wave_B_16k.shape[-1])
    # print("hubert_A_mask", hubert_A_mask.shape)
    # print("hubert_B_mask", hubert_B_mask.shape)

    assert_tensors(hubert_A_mask, hubert_B_mask, "hubert_A_mask", "hubert_B_mask")

    # Check passes hubert correctly
    hubert_feats_A_16k, hubert_A_mask_out = hubert(wave_A_16k, hubert_A_mask)
    hidden_states_A = torch.stack(hubert_feats_A_16k.hidden_states).transpose(-1, -2)
    hubert_feats_B_16k, hubert_B_mask_out = hubert(wave_B_16k, hubert_B_mask)
    hidden_states_B = torch.stack(hubert_feats_B_16k.hidden_states).transpose(-1, -2)
    # print("hidden_states")
    # print(hubert_feats_A_16k.shape)
    # print(hubert_feats_B_16k.shape)
    # print(hidden_states_A.shape)
    # print(hidden_states_B.shape)
    # print(hubert_A_mask_out.shape)
    # print(hubert_B_mask_out.shape)
    # assert_tensors(hubert_feats_A_16k.last_hidden_state.transpose(-1, -2), hubert_feats_B_16k.last_hidden_state.transpose(-1, -2), "hubert_feats_A_16k", "hubert_feats_B_16k")
    
    # assert_tensors(hubert_feats_A_16k, hubert_feats_B_16k, "extract_features_A", "extract_features_B")
    assert_tensors(hidden_states_A * hubert_A_mask_out, hidden_states_B * hubert_B_mask_out, "hubert_feats_A_16k", "hubert_feats_B_16k")
    assert_tensors(hubert_A_mask_out, hubert_B_mask_out, "hubert_A_mask_out", "hubert_B_mask_out")

    # Check passes pooling module
    pooling_feats_A_16k = pooling(hubert_feats_A_16k) * hubert_A_mask_out.unsqueeze(2)
    pooling_feats_B_16k = pooling(hubert_feats_B_16k) * hubert_B_mask_out.unsqueeze(2)
    # print(pooling_feats_A_16k.shape)
    # print(pooling_feats_B_16k.shape)
    assert_tensors(pooling_feats_A_16k.transpose(-1, -2), pooling_feats_B_16k.transpose(-1, -2), "pooling_feats_A_16k", "pooling_feats_B_16k")

    # Create 24k waves
    wave_A_24k = resampler_48k_to_24k(wave_A_48k)
    wave_B_24k = resampler_48k_to_24k(wave_B_48k)
    # print(wave_A_24k.shape)
    # print(wave_B_24k.shape)
    assert_tensors(wave_A_24k, wave_B_24k, "wave_A_24k", "wave_B_24k")

    # Create target mels
    target_mel_A = ap.get_mel_spectrogram(wave_A_24k)
    target_mel_B = ap.get_mel_spectrogram(wave_B_24k)
    # print(target_mel_A.shape)
    # print(target_mel_B.shape)
    assert_tensors(target_mel_A, target_mel_B, "target_mel_A", "target_mel_B", pad_value=torch.log(torch.tensor(1e-7)).item())

    # Create mel masks
    mel_mask_A = mask_manager.get_mel_mask(lens_48k_A, target_mel_A.shape[-1])
    mel_mask_B = mask_manager.get_mel_mask(lens_48k_B, target_mel_B.shape[-1])
    # print(mel_mask_A.shape)
    # print(mel_mask_B.shape)
    assert_tensors(mel_mask_A, mel_mask_B, "mel_mask_A", "mel_mask_B")

    # Check passes adapter
    # print(pooling_feats_A_16k.shape)
    # print(pooling_feats_B_16k.shape)
    predicted_mel_A = adapter(pooling_feats_A_16k, target_mel_A.shape[-1], mask=hubert_A_mask_out, mel_mask=mel_mask_A)
    predicted_mel_B = adapter(pooling_feats_B_16k, target_mel_B.shape[-1], mask=hubert_B_mask_out, mel_mask=mel_mask_B)
    # print(predicted_mel_A.shape)
    # print(predicted_mel_B.shape)
    assert_tensors(predicted_mel_A , predicted_mel_B , "predicted_mel_A", "predicted_mel_B")

    # Check criterion
    # print("In Loss")
    loss_A = criterion(predicted_mel_A, target_mel_A) * mel_mask_A
    loss_B = criterion(predicted_mel_B, target_mel_B) * mel_mask_B
    assert_tensors(loss_A, loss_B, "loss_A", "loss_B")

    num_active_A = mel_mask_A.sum() * predicted_mel_A.shape[1]
    num_active_B = mel_mask_B.sum() * predicted_mel_B.shape[1]
    assert num_active_A == num_active_B

    loss_A = loss_A.sum() / num_active_A
    loss_B = loss_B.sum() / num_active_B
    # print(loss_A, loss_B)
    assert torch.allclose(loss_A, loss_B, atol=1e-7)
