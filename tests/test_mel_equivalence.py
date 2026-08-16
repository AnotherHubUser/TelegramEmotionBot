import pytest
import torch
from vocos import Vocos
from utils.audio import AudioProcessor

def test_mel_equivalence():
    """Check AudioProcessor returns mel identical to origin Vocos"""
    torch.manual_seed(42)
    waveform = torch.randn(1, 72000)
    
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    with torch.no_grad():
        mel_vocos = vocos.feature_extractor(waveform)
        
    audio_processor = AudioProcessor()
    mel_manual_log = audio_processor.get_mel_spectrogram(waveform)
    
    assert mel_vocos.shape == mel_manual_log.shape, \
        f"Different matrix shape! {mel_vocos.shape} vs {mel_manual_log.shape}"        
    assert torch.allclose(mel_vocos, mel_manual_log, atol=1e-5), \
        f"mel-grams doesn't coinside! Max diff: {torch.abs(mel_vocos - mel_manual_log).max().item():.8f}"
