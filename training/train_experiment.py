import torch
import torchaudio
from utils.audio import AudioProcessorWrapper
from models.hubert import HubertWrapper
from models.vocoder import VocoderWrapper
from pathlib import Path
from config.identity_config import TrainingConfig
from runners.identity_runner import IdentityRunner

def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6

def inference_single_wave(wave_path: str, output_path: str, checkpoint_filename: str):
    config = TrainingConfig()

    class InferenceIdentityRunner(IdentityRunner):
        def _build_dataloader(self):
            return None
    runner = InferenceIdentityRunner(config, cache_dir="", run_preprocess=False)
    
    checkpoint_path = Path(config.checkpoint_dir) / checkpoint_filename
    runner.trainer.load_checkpoint(str(checkpoint_path))
    
    adapter = runner.trainer.adapter.eval()
    pooling = runner.trainer.pooling.eval()
    print(f"Параметры Адаптера: {count_parameters(adapter):.2f} M")
    print(f"Параметры Пулинга: {count_parameters(pooling):.6f} M")

    
    weights = torch.nn.functional.softmax(pooling.layer_weights, dim=0)
    print(weights)
    
    wave, sr = torchaudio.load(wave_path)
    
    mel_processor = AudioProcessorWrapper(config)
    feature_extractor = HubertWrapper(config)
    vocoder = VocoderWrapper(config)
    
    with torch.no_grad():
        target_mel = mel_processor(wave, sr).squeeze().to(config.device)  # [100, T_mel]
        target_len = target_mel.shape[-1]
        hubert_feats_layers = feature_extractor(wave, sr).squeeze().to(config.device) # [13, T_feats, 768]
        
        hubert_feats_layers = hubert_feats_layers.unsqueeze(0) # [1, 13, T_feats, 768]
        
        pooling_feats = pooling(hubert_feats_layers) # [1, T_feats, 768]
        predicted_mel = adapter(pooling_feats, target_len, mask=None, mel_mask=None) # [1, 100, T_mel]
        output_wave = vocoder(predicted_mel).squeeze() # [100, T_mel]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, output_wave.cpu(), config.vocoder_sr)
        print(f"Готово! Результат сохранен в {output_path}")

if __name__ == "__main__":
    inference_single_wave(
        wave_path="data/train/anya/sample1.ogg",
        # wave_path="archive/RAVDESS/Actor_02/03-01-03-01-01-02-02.wav",
        output_path="data/train/anya_output/sample1_ep0.ogg",
        checkpoint_filename="checkpoint_epoch_0.pt",
        # checkpoint_filename="onebatched_10049.pt"
    )
