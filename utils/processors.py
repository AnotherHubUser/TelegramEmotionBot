from abc import ABC, abstractmethod
import torch

class AudioToEmbeddingsProcessor(ABC):
    @abstractmethod
    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Transforms wave and sr to embeddings"""
        pass
    
class AudioToMelProcessor(ABC):
    @abstractmethod
    def __call__(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Transforms wave and sr to Mel-spectrogram"""
        pass

class MelToAudioProcessor(ABC):
    @abstractmethod
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """Transforms Mel-spectrogram to waveform"""
        pass
