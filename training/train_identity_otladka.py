import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.adapter import EmotionAdapter
from models.hubert import HubertEmbeddings
from utils.audio import AudioProcessor
from utils.dataset import RAVDESSDataset, collate_fn
from models.vocoder import Vocoder
from models.learnable_layer_pooling import LearnableLayerPooling

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="An output with one or more elements was resized")


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    
    dataset = RAVDESSDataset('archive', target_sr=48000)
    ap = AudioProcessor().to(device)
    vocoder = Vocoder().to(device)
    hubert = HubertEmbeddings().to(device)    
    num_layers = hubert.get_num_layers()
    pooling_module = LearnableLayerPooling(num_layers=num_layers).to(device)
    
    resampler_48k_to_24k = torchaudio.transforms.Resample(48000, 24000).to(device)
    resampler_48k_to_16k = torchaudio.transforms.Resample(48000, 16000).to(device)
    
    model = EmotionAdapter().to(device) # HERE
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 2e-4},
        {'params': pooling_module.parameters(), 'lr': 3e-4}
        ])
    criterion = nn.L1Loss()

    model.train()
    pooling_module.train()

    # waveform, _, _ = dataset[19] # [1, L_48k]
    # waveform = waveform.to(device)
    # print("waveform:\t", waveform.shape)
        
    # with torch.no_grad():
    #     wave_24k = resampler_48k_to_24k(waveform) # [1, L_24k]
    #     wave_16k = resampler_48k_to_16k(waveform) # [1, L_16k]
    #     target_mel = ap.get_mel_spectrogram(wave_24k) # [1, 100, T_mel]
    #     target_len = target_mel.shape[-1]
    #     hubert_feats = hubert(wave_16k) # [num_layers, T, 768]
    #     hubert_feats = pooling_module(hubert_feats) # [1, T, 768]


    # print(f"target_mel\t min:{target_mel.min().item():.2f}\t max{target_mel.max().item():.2f}")

    # chosen_batch = [19, 25, 31, 67]
    batch = next(iter(DataLoader(dataset, batch_size=4, collate_fn=collate_fn)))
    waveform = batch["anchor"]
    waveform = waveform.to(device)
        
    with torch.no_grad():
        wave_24k = resampler_48k_to_24k(waveform)
        wave_16k = resampler_48k_to_16k(waveform)
        target_mel = ap.get_mel_spectrogram(wave_24k) # [B, 100, T_mel]
        target_len = target_mel.shape[-1]
        hubert_feats_layers = hubert(wave_16k) # [B, T, 768]

    for epoch in tqdm(range(10011)): 
        optimizer.zero_grad()
        hubert_feats = pooling_module(hubert_feats_layers)
        predicted_mel = model(hubert_feats, target_len)
        loss = criterion(predicted_mel, target_mel)
        loss.backward()
        
        if epoch % 1000 == 10:
            print(f"predicted_mel\t min:{predicted_mel.min().item():.2f}\t max{predicted_mel.max().item():.2f}")
            first_layer_grad = model.input_proj.weight.grad
            if first_layer_grad is not None:
                print(f"\n[Epoch {epoch}] Норма градиента 1-го слоя: {first_layer_grad.norm().item():.6f}")
            else:
                print(f"\n[Epoch {epoch}] Градиенты не дошли до слоя!")
            print(first_layer_grad.type(), first_layer_grad.shape)

        weights_before = model.input_proj.weight.data.clone()
        
        optimizer.step()

        if epoch % 1000 == 10:
            weight_delta = torch.abs(weights_before - model.input_proj.weight.data).sum().item()
            print(model.input_proj.weight.data.type(), model.input_proj.weight.data.shape)
            print(f"[Epoch {epoch}] Сдвиг весов оптимизатором: {weight_delta:.6f}")
            print(f"[Epoch {epoch}] Реальный Лосс: {loss.item():.6f}")

    
    # waveform, _, _ = dataset[79]
    waveform = waveform[2][:batch["anchor_lens"][2]].unsqueeze(dim=0)
    waveform = waveform.to(device)
        
    model.eval()
    pooling_module.eval()

    with torch.no_grad():
        wave_24k = resampler_48k_to_24k(waveform)
        wave_16k = resampler_48k_to_16k(waveform)
        target_mel = ap.get_mel_spectrogram(wave_24k) # [1, 100, T_mel]
        target_len = target_mel.shape[-1]
        hubert_feats_layers = hubert(wave_16k) # [1, T, 768]
        
    hubert_feats = pooling_module(hubert_feats_layers)
    predicted_mel = model(hubert_feats, target_len)
    print(f"{criterion(predicted_mel, target_mel).item():6f}")
        
    with torch.no_grad():
        normal_predict = model(hubert_feats, target_len)
        normal_loss = criterion(normal_predict, target_mel).item()
        print(f"Лосс на реальном HuBERT: {normal_loss:.6f}")
        
        noise_feats = torch.randn_like(hubert_feats)
        noise_predict = model(noise_feats, target_len)
        noise_loss = criterion(noise_predict, target_mel).item()
        print(f"Лосс на СЛУЧАЙНОМ ШУМЕ: {noise_loss:.6f}")
        
        noise_audio = vocoder.generate(noise_predict)
        torchaudio.save("data/train/identity_reconstruction/noise.wav", noise_audio.cpu(), 24000)

    with torch.no_grad():
        target_audio = vocoder.generate(target_mel)
        final_audio = vocoder.generate(predicted_mel)
        torchaudio.save("data/train/identity_reconstruction/target.wav", target_audio.cpu(), 24000)
        torchaudio.save("data/train/identity_reconstruction/test.wav", final_audio.cpu(), 24000)
        print("CHECK THIS OUT")


if __name__ == "__main__":
    train()
