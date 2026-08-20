import torch


class TrainingConfig:
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device: str = torch.device("mps")
    num_workers = 4
    seed: int = 29
    
    RAVDESS_dataset_dir: str = "archive/RAVDESS"
    # LibriTTS_R_dataset_dir: str = "archive/LibriTTS_R"
    LibriTTS_R_dataset_dir: str = "/kaggle/input/datasets/justsahil/libritts-r"
    checkpoint_dir: str = "checkpoints/identity_reconstruction"
    output_dir: str = "data/train/identity_reconstruction"

    mel_rel_dir: str = "target_mel"
    feats_rel_dir: str = "features_embeddings"
    
    batch_wave_key: str = "wave"
    batch_sr_key: str = "sr"
    batch_path_key: str = "relative_path"
    batch_mel_key: str = "target_mel"
    batch_mel_mask_key: str = "target_mel_mask"
    batch_feats_key: str = "feats_outputs"
    batch_feats_mask_key: str = "feats_outputs_mask"

    models_pooling_name: str = "learnable_pooling"
    models_adapter_name: str = "adapter"

    snapshot_epoch_key: str = "epoch"
    snapshot_loss_key: str = "loss"
    snapshot_models_state_key: str = "models_state_dict"
    snapshot_optimizer_state_key: str = "optimizer_state_dict"

    step_loss_key: str = "loss"
    step_loss_noise_key: str = "loss"
    step_grad_norm_key: str = "grad_norm"
    step_total_grad_norm_key: str = "total_grad_norm"
    step_predicted_mel_key: str = "predicted_mel"
    step_target_mel_key: str = "target_mel"
    step_mel_mask_key: str = "mel_mask"

    orig_sr: int = 48000

    feats_model_name: str = "facebook/hubert-base-ls960"
    # feats_model_name: str = "./hubert_local"
    feats_sr: int = 16000
    feats_output_dim: int = 768

    vocoder_model_name: str = "charactr/vocos-mel-24khz"
    vocoder_sr: int = 24000
    vocoder_hop_length: int = 256
    vocoder_n_fft: int = 1024
    vocoder_input_dim: int = 100
    
    adapter_hidden_dim: int = 256

    batch_size: int = 16
    epochs: int = 20
    
    adapter_lr: float = 2e-4
    pooling_lr: float = 3e-4
    max_grad_norm: float = 1.0


class DebugConfig(TrainingConfig):
    """fast one-butch check"""
    epochs: int = 10050
    batch_size: int = 4 # 32 leader
    save_outputs: bool = True
