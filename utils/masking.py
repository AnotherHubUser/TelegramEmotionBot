import torch


class MaskManager:
    def __init__(self, orig_sr=48000, hubert_sr=16000, vocos_sr=24000, hop_length=256):
        self.orig_sr = orig_sr
        self.hubert_sr = hubert_sr
        self.vocos_sr = vocos_sr
        self.hop_length = hop_length
        self.to_hubert_coef = orig_sr // hubert_sr
        self.to_vocos_coef = orig_sr // vocos_sr

    def _resample_lens(self, lens: torch.Tensor, coef: int) -> torch.Tensor:
        """resamples wav lens with origin sr to new sr according coef = or_sr // new_sr"""
        return (lens - 1) // coef + 1

    def create_binary_mask(self, padding_len: int, lens: torch.Tensor) -> torch.Tensor:
        """Generate 2D binary mask [B, T_max]"""
        grid = torch.arange(padding_len, device=lens.device)[None, :]
        return (grid < lens[:, None]).to(torch.int32)

    def get_hubert_wav_mask(self, orig_lens: torch.Tensor, max_len_16k: int) -> torch.Tensor:
        """Get mask of wav for HuBERT [B, L_16k_max]"""
        lens_16k = self._resample_lens(orig_lens, self.to_hubert_coef)
        return self.create_binary_mask(max_len_16k, lens_16k)

    def get_mel_mask(self, orig_lens: torch.Tensor, max_mel_frames: int) -> torch.Tensor:
        """Get mask of mel-spec. for Vocos [B, T_mel_max]"""
        lens_24k = self._resample_lens(orig_lens, self.to_vocos_coef)
        mel_lens = lens_24k // self.hop_length + 1
        return self.create_binary_mask(max_mel_frames, mel_lens)


def create_binary_mask(padding_len: int, lens: torch.Tensor) -> torch.Tensor:
    """Generate 2D binary mask [B, T_max]"""
    grid = torch.arange(padding_len, device=lens.device)[None, :]
    return (grid < lens[:, None]).to(torch.long)
