from utils.dataset import RAVDESSDataset
from models.hubert import get_hubert_embeddings
from utils.audio import AudioProcessor
from models.vocoder import Vocoder
import torchaudio
import torch.nn.functional as F
import torch
from vocos import Vocos

# 1. Загружаем аудио (твой датасет)
dataset = RAVDESSDataset('archive')
waveform, label, _ = dataset[27] # waveform: [1, T]
waveform_resampled = torchaudio.transforms.Resample(16000, 24000)(waveform)

# waveform_resampled = waveform_resampled[:, :2559]
print(f"Waveform:\t{waveform_resampled.shape}")
print(f"Label:\t\t{label}")

vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
# mel_spec = vocos.feature_extractor(waveform_resampled)
# output_audio = vocos.decode(mel_spec)
audio_processor = AudioProcessor()
mel_spec_manual = audio_processor.get_mel_spectrogram(waveform_resampled)
# len = waveform_resampled.shape[1] // 256 + 1
output_audio_manual = vocos.decode(mel_spec_manual)


resampler = torchaudio.transforms.Resample(16000, 24000)
dummy_adapter = torch.nn.Linear(768, 100)
its = 0
print(len(dataset))
for waveform, label, _ in dataset:
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0).unsqueeze(0)
    its += 1
    waveform_res = resampler(waveform)
    print("waveforms shape")
    print(waveform.shape)
    print(waveform_res.shape)
    target_mel_spec = audio_processor.get_mel_spectrogram(waveform_res)
    
    hubert_embeds = get_hubert_embeddings(waveform)
    dummy_mel_spec = dummy_adapter(hubert_embeds).transpose(1, 2)
    dummy_mel_spec = F.interpolate(dummy_mel_spec, size=target_mel_spec.shape[-1])

    assert dummy_mel_spec.shape == target_mel_spec.shape, f"{dummy_mel_spec.shape} != {target_mel_spec.shape}"

print(f"all assertions {its} passed")

    

# # Представь, что мы получили эмбеддинги: [1, T_hubert, 768]
# hubert_embeds = get_hubert_embeddings(waveform[:, :8079])
hubert_embeds = get_hubert_embeddings(waveform)
# len = (waveform.shape[1] - 80) // 320
print("hubert", hubert_embeds.shape)
# # Просто линейный слой, который сжимает 768 признаков в 80 (Mel-bands)
dummy_adapter = torch.nn.Linear(768, 100)
# # Нам нужно поменять оси для HiFi-GAN: [1, T, 80] -> [1, 80, T]
dummy_mel_spec = dummy_adapter(hubert_embeds)
print("dummy mel", dummy_mel_spec.shape)
dummy_mel_spec = F.interpolate(dummy_mel_spec.transpose(1, 2), size=mel_spec_manual.shape[-1])

print("dummy mel", dummy_mel_spec.shape)

# mel_spec = AudioProcessor().get_mel_spectrogram(waveform) 
# vocoder = Vocoder()
# output_audio = vocoder.generate(mel_spec)

# print(f"Original shape:\t{waveform.shape}")
# print(f"Ideal MelShape:\t{mel_spec.shape}")
print(f"Manual MelShape:\t{mel_spec_manual.shape}")
# print(f"Output2 shape:\t{output_audio.shape}")
print(f"Output2 shape:\t{output_audio_manual.shape}")

# 5. Сохраняем результат
# import torchaudio
# torchaudio.save("data/train/original.wav", waveform.cpu(), 16000)
# torchaudio.save("data/train/test_reconstruction.wav", output_audio.cpu(), 24000)
# torchaudio.save("data/train/test_reconstruction_manual.wav", output_audio_manual.cpu(), 24000)
