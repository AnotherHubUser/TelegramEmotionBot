import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from tqdm import tqdm

from models.adapter import EmotionAdapter
from models.hubert import HubertEmbeddings
from utils.audio import AudioProcessor
from utils.dataset import RAVDESSDataset
from models.vocoder import Vocoder
from models.learnable_layer_pooling import LearnableLayerPooling

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="An output with one or more elements was resized")


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    
    dataset = RAVDESSDataset('archive')
    ap = AudioProcessor().to(device)
    vocoder = Vocoder().to(device)
    hubert = HubertEmbeddings().to(device)
    num_layers = hubert.get_num_layers()
    pooling_module = LearnableLayerPooling(num_layers=num_layers).to(device)
    resampler_16k_to_24k = torchaudio.transforms.Resample(16000, 24000).to(device)
    
    model = EmotionAdapter().to(device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': 2e-4},
        {'params': pooling_module.parameters(), 'lr': 3e-4}
        ], lr=1e-3)
    
    criterion = nn.L1Loss()

    test_sample2_path = "data/train/anya/sample2.ogg"
    test_wave = dataset._load_audio(test_sample2_path)
    test_wave = test_wave.to(device)
    wave_res = resampler_16k_to_24k(test_wave)
    test_target_mel = ap.get_mel_spectrogram(wave_res)
    test_hubert_feats = hubert(test_wave)
    
    test_hubert_pools = pooling_module(test_hubert_feats)
    test_predicted_mel = model(test_hubert_pools, test_target_mel.shape[-1])
        
    model.train()
    
    for epoch in range(10):
        epoch_loss = 0
        for waveform, label, _ in tqdm(dataset):
            # waveform, _, _ = dataset[19]
            waveform = waveform.to(device)
            # print(waveform.shape)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0).unsqueeze(0)
            # print(waveform.shape)
            # print(waveform.shape)
            with torch.no_grad():
                # 16k -> 24k
                wave_24k = resampler_16k_to_24k(waveform)
                target_mel = ap.get_mel_spectrogram(wave_24k) # [1, 100, T_mel]
                target_len = target_mel.shape[-1]
                hubert_outputs = hubert(waveform) # [1, T, 768]
            
            optimizer.zero_grad()
            
            hubert_feats = pooling_module(hubert_outputs)
            predicted_mel = model(hubert_feats, target_len)

            loss = criterion(predicted_mel, target_mel)
            loss.backward()

            epoch_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch} | Loss: {epoch_loss/len(dataset):.6f}")
        print(torch.softmax(pooling_module.layer_weights, dim=0))

        model.eval()
        with torch.no_grad():
            test_hubert_pools = pooling_module(test_hubert_feats)
            test_predicted_mel = model(test_hubert_pools, test_target_mel.shape[-1])
            print(f"{criterion(test_predicted_mel, test_target_mel).item():6f}")

        model.train()
    
    model.eval()
    
    # wave, _, _ = dataset[19]
    test_sample2_path = "data/train/anya/sample2.ogg"
    wave = dataset._load_audio(test_sample2_path)
    wave = wave.to(device)
    print("SDFS", wave.shape)
    wave_res = resampler_16k_to_24k(wave)
    target_mel = ap.get_mel_spectrogram(wave_res)
    hubert_feats = hubert(wave)
    hubert_feats = pooling_module(hubert_feats)
    predicted_mel = model(hubert_feats, target_mel.shape[-1])
    print(f"{criterion(predicted_mel, target_mel).item():6f}")
        
    with torch.no_grad():
        target_audio = vocoder.generate(target_mel)
        final_audio = vocoder.generate(predicted_mel)
        torchaudio.save("data/train/anya_output/sample2_target.wav", target_audio.cpu(), 24000)
        torchaudio.save("data/train/anya_output/sample2_predicted.wav", final_audio.cpu(), 24000)
        print("CHECK THIS OUT")


if __name__ == "__main__":
    train()
