# import torch
# import torchaudio
# import time
import sys
import os

print("================= KAGGLE ENVIRONMENT INSPECTION =================")
print(f"Current Working Directory (cwd): {os.getcwd()}")
print(f"Absolute path of this script (__file__): {os.path.abspath(__file__)}")
print(f"Python search paths (sys.path): {sys.path}")

print("\n--- Listing contents of /kaggle/ ---")
if os.path.exists("/kaggle"):
    print(os.listdir("/kaggle"))
else:
    print("/kaggle directory does not exist")

print("\n--- Listing contents of /kaggle/working/ ---")
if os.path.exists("/kaggle/working"):
    print(os.listdir("/kaggle/working"))
else:
    print("/kaggle/working directory does not exist")

print("\n--- Listing contents of current script directory ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
print(os.listdir(script_dir))
print("================================================================")

# Принудительно завершаем скрипт, чтобы не падать на ошибках импорта дальше
sys.exit(0)

import multiprocessing as mp

# from torch.utils.data import DataLoader
from config.identity_config import TrainingConfig, DebugConfig
# from datasets.LibriTTS_R_dataset import LibriTTSRDataset
# from datasets.precomputed_identity_dataset import PrecomputedIdentityDataset, make_identity_cached_collate_fn
# from models.hubert import HubertWrapper
# from models.adapter import EmotionAdapter
# from models.vocoder import VocoderWrapper
# from models.learnable_layer_pooling import LearnableLayerPooling
# from trainers.identity_trainer import IdentityTrainer
# from pipelines.preprocess_identity import IdentityPreprocessPipeline
# from utils.audio import AudioProcessorWrapper
# from utils.dataset import RAVDESSDataset, collate_fn
# from tqdm import tqdm
from runners.identity_runner import IdentityRunner, OnebatchIdentityRunner, OnlineIdentityRunner

import warnings
# Глушим конкретное предупреждение про resize в STFT
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

# def preprocess():
#     config = DebugConfig()
#     dirs = ["./archive/LibriTTS_R/dev-clean"]
#     for dataset_dir in dirs:
#         dataset = LibriTTSRDataset(config, dataset_dir)
#         mel_processor = AudioProcessorWrapper(config) 
#         feature_extractor = HubertWrapper(config)
#         output_dir = dataset_dir
        
#         pipeline = IdentityPreprocessPipeline(
#             config=config,
#             dataset=dataset,
#             mel_processor=mel_processor,
#             feature_extractor=feature_extractor,
#             output_dir=output_dir
#         )
#         pipeline.run()
#         print(f"mel resemplers\t{mel_processor.resemplers}")
#         print(f"extractor resemplers\t{feature_extractor.resemplers}")

# def train():
#     dir = "./archive/LibriTTS_R/dev-clean"

#     config = DebugConfig()
#     dataset = PrecomputedIdentityDataset(config, cache_dir=dir)
#     collate_fn = make_identity_cached_collate_fn(config)
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=config.batch_size, 
#         shuffle=True, 
#         num_workers=config.num_workers,
#         collate_fn=collate_fn,
#     )

#     # dataset = RAVDESSDataset(config.RAVDESS_dataset_dir, target_sr=config.orig_sr)

#     adapter = EmotionAdapter(
#         input_dim=config.hubert_output_dim, 
#         hidden_dim=config.adapter_hidden_dim, 
#         output_dim=config.vocoder_input_dim).to(config.device)

#     num_layers = HubertWrapper(config).model.get_num_layers()   
#     pooling = LearnableLayerPooling(num_layers=num_layers).to(config.device)
#     models = {
#         config.models_adapter_name: adapter,
#         config.models_pooling_name: pooling, 
#     }
    
#     optimizer = torch.optim.Adam([
#         {'params': adapter.parameters(), 'lr': config.adapter_lr},
#         {'params': pooling.parameters(), 'lr': config.pooling_lr}
#     ])
#     criterion = torch.nn.L1Loss(reduction='none')
#     trainer = IdentityTrainer(config, models, optimizer, criterion)

#     batch = next(iter(DataLoader(dataset, batch_size=config.batch_size, collate_fn=collate_fn)))

#     # hubert_outputs, hubert_mask_out = trainer.get_hubert_feats(batch)

#     for epoch in tqdm(range(config.epochs)):
#         # print(f"epoch {epoch} time: {time.perf_counter() - time_before_step:.4f} secs.")

#         # trainer._check_hubert_step(batch)
#         step_data = trainer.train_step(batch)
        
#         if epoch % 1000 == 1:
#             print(f"\n[Epoch {epoch}] Actual Loss: {step_data[config.step_loss_key]:.6f}")
#             print(f"Grad norm 1st layer: {step_data[config.step_grad_norm_key]:.6f}")
#             print(f"Grad norm total: {step_data[config.step_total_grad_norm_key]:.6f}")
            
#     sanity = trainer.evaluate_sanity(batch)

#     vocoder = VocoderWrapper(config)
#     with torch.no_grad():
#         target_audio = vocoder(sanity[config.step_target_mel_key])[-1]
#         predicted_audio = vocoder(sanity[config.step_predicted_mel_key])[-1]
#         torchaudio.save("data/train/identity_reconstruction/target.wav", target_audio.cpu(), 24000)
#         torchaudio.save("data/train/identity_reconstruction/test.wav", predicted_audio.cpu(), 24000)

#     print(f"Лосс на реальном HuBERT: {sanity[config.step_loss_key]:.6f}")
#     print(f"Лосс на СЛУЧАЙНОМ ШУМЕ: {sanity[config.step_loss_noise_key]:.6f}")

def main():
    # mode = "debug"
    mode = "training"
    # if torch.backends.mps.is_available():
    #     mp.set_start_method("forkserver", force=True)
    mp.set_start_method("fork", force=True)

    if mode == "debug":
        config = DebugConfig()
        cache_dir = "./archive/LibriTTS_R/dev-clean"
        runner = OnebatchIdentityRunner(config, cache_dir, False)
    else:
        config = TrainingConfig()
        cache_dir = "./archive/LibriTTS_R/dev-clean"
        # runner = IdentityRunner(config, cache_dir, False)
        runner = OnlineIdentityRunner(config, cache_dir)

    runner.run()


if __name__ == "__main__":
    # preprocess()
    # train()
    main()