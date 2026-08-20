# pipelines/debug_runner.py
import torch
import torchaudio
from torch.utils.data import DataLoader
import time
from pathlib import Path
from tqdm import tqdm
from config.identity_config import TrainingConfig
from runners.base_runner import BaseRunner
from models.adapter import EmotionAdapter
from models.learnable_layer_pooling import LearnableLayerPooling
from models.hubert import HubertWrapper
from models.vocoder import VocoderWrapper
from pipelines.preprocess_identity import IdentityPreprocessPipeline
from trainers.identity_trainer import IdentityTrainer, IdentityFullTrainer
from datasets.precomputed_identity_dataset import PrecomputedIdentityDataset, make_identity_cached_collate_fn
from datasets.LibriTTS_R_dataset import LibriTTSRDataset
from utils.audio import AudioProcessorWrapper
        

class IdentityRunner(BaseRunner):
    def _execute_preprocessing(self):        
        raw_dataset = LibriTTSRDataset(self.config, self.cache_dir)
        mel_processor = AudioProcessorWrapper(self.config) 
        feature_extractor = HubertWrapper(self.config)

        pipeline = IdentityPreprocessPipeline(
            config=self.config,
            dataset=raw_dataset,
            mel_processor=mel_processor,
            feature_extractor=feature_extractor,
            output_dir=self.cache_dir
        )
        pipeline.run()
        
        del mel_processor, feature_extractor, pipeline, raw_dataset

    def _build_dataloader(self) -> DataLoader:
        self.dataset = PrecomputedIdentityDataset(self.config, cache_dir=self.cache_dir)
        self.collate_fn = make_identity_cached_collate_fn(self.config)
        return DataLoader(
            self.dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers,
            # num_workers=0,
            collate_fn=self.collate_fn,
        )

    def _build_trainer(self) -> IdentityTrainer:
        adapter = EmotionAdapter(
            input_dim=self.config.feats_output_dim, 
            hidden_dim=self.config.adapter_hidden_dim, 
            output_dim=self.config.vocoder_input_dim
        ).to(self.config.device)

        num_layers = HubertWrapper(self.config).model.get_num_layers()   
        pooling = LearnableLayerPooling(num_layers=num_layers).to(self.config.device)
        
        is_multi_gpu = 'cuda' in str(self.config.device) and torch.cuda.device_count() > 1
        class TransparentDataParallel(torch.nn.DataParallel):
            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

            def parameters(self, recurse=True):
                return self.module.parameters(recurse=recurse)

            def state_dict(self, *args, **kwargs):
                return self.module.state_dict(*args, **kwargs)

            def load_state_dict(self, state_dict, strict=True):
                return self.module.load_state_dict(state_dict, strict=strict)

            
        if is_multi_gpu:
            adapter = TransparentDataParallel(adapter)
            pooling = TransparentDataParallel(pooling)
            
        models = {
            self.config.models_adapter_name: adapter,
            self.config.models_pooling_name: pooling, 
        }
        
        optimizer = torch.optim.Adam([
            {'params': adapter.parameters(), 'lr': self.config.adapter_lr},
            {'params': pooling.parameters(), 'lr': self.config.pooling_lr}
        ])
        criterion = torch.nn.L1Loss(reduction='none')
        
        return IdentityTrainer(self.config, models, optimizer, criterion)

    def run(self, checkpoint=None):
        start_epoch = 0
        if checkpoint is not None:
            checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint
            if checkpoint_path.exists():
                start_epoch, _ = self.trainer.load_checkpoint(checkpoint_path)

        start_time = time.perf_counter()

        for epoch in range(start_epoch, self.config.epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
            print(f"loaded progress bar {time.perf_counter() - start_time:.4f} secs.")
            start_time = time.perf_counter()
            for batch in progress_bar:
                print(f"loaded batch in {time.perf_counter() - start_time}")
                start_time = time.perf_counter()
                step_data = self.trainer.train_step(batch)
                print(f"trainer step in {time.perf_counter() - start_time}")
                loss_val = step_data[self.config.step_loss_key]
                epoch_loss += loss_val
                
                progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})
                start_time = time.perf_counter()
                
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"[Epoch {epoch} completed] Avg loss: {avg_loss:.6f}")
            
            weights = torch.nn.functional.softmax(self.trainer.pooling.layer_weights, dim=0)
            print(weights)
            
            # if epoch % 5 == 0 or epoch == self.config.epochs - 1:
            self.trainer.save_checkpoint(
                epoch=epoch,
                loss=avg_loss,
                filename=f"checkpoint_{Path(self.cache_dir).name}_epoch_{epoch}.pt"
            )


class OnebatchIdentityRunner(IdentityRunner):
    def run(self):
        batch = next(iter(self.dataloader))

        for epoch in tqdm(range(self.config.epochs), desc="Overfitting"):
            step_data = self.trainer.train_step(batch)
            
            if epoch % 1000 == 1:
                print(f"\n[Epoch {epoch}] Actual Loss: {step_data[self.config.step_loss_key]:.6f}")
                print(f"Grad norm 1st layer: {step_data[self.config.step_grad_norm_key]:.6f}")
                print(f"Grad norm total: {step_data[self.config.step_total_grad_norm_key]:.6f}")
                
        sanity = self.trainer.evaluate_sanity(batch)
        print(f"Loss on actual feats: {sanity[self.config.step_loss_key]:.6f}")
        print(f"Loss on noise: {sanity[self.config.step_loss_noise_key]:.6f}")

        vocoder = VocoderWrapper(self.config)
        with torch.no_grad():
            target_audio = vocoder(sanity[self.config.step_target_mel_key])[-1]
            predicted_audio = vocoder(sanity[self.config.step_predicted_mel_key])[-1]
            
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torchaudio.save(out_dir / "target.wav", target_audio.cpu(), self.config.vocoder_sr)
            torchaudio.save(out_dir / "test.wav", predicted_audio.cpu(), self.config.vocoder_sr)
        
        self.trainer.save_checkpoint(
            epoch=self.config.epochs,
            loss=sanity[self.config.step_loss_key],
            filename=f"onebatched_{epoch}.pt"
        )


class OnlineIdentityRunner(IdentityRunner):
    def __init__(self, config: TrainingConfig, cache_dir: str):
        super().__init__(config, cache_dir, run_preprocess=False)
        self.mel_processor = AudioProcessorWrapper(self.config)
        self.feature_extractor = HubertWrapper(self.config)
        self.identity_cached_collate_fn = make_identity_cached_collate_fn(self.config)

    def _execute_preprocessing(self):
        pass

    def _build_trainer(self) -> IdentityTrainer:
        adapter = EmotionAdapter(
            input_dim=self.config.feats_output_dim, 
            hidden_dim=self.config.adapter_hidden_dim, 
            output_dim=self.config.vocoder_input_dim
        ).to(self.config.device)

        num_layers = HubertWrapper(self.config).model.get_num_layers()   
        pooling = LearnableLayerPooling(num_layers=num_layers).to(self.config.device)
        
        models = {
            self.config.models_adapter_name: adapter,
            self.config.models_pooling_name: pooling, 
        }
        
        optimizer = torch.optim.Adam([
            {'params': adapter.parameters(), 'lr': self.config.adapter_lr},
            {'params': pooling.parameters(), 'lr': self.config.pooling_lr}
        ])
        criterion = torch.nn.L1Loss(reduction='none')        
        return IdentityFullTrainer(self.config, models, optimizer, criterion)


    def _build_dataloader(self) -> DataLoader:
        self.raw_dataset = LibriTTSRDataset(self.config, self.cache_dir)
        
        return DataLoader(
            self.raw_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers,
            collate_fn=self._online_collate_fn,
        )
    
    def _online_collate_fn(self, batch_list):
        # Вход список словарей {wave: [1, wave_len], sr: int}
        # Выход словарь 4 батча {
        #   feats:      [B, 13, T_feat_max, 768],
        #   feats_mask: [B, T_feat_max],
        #   mel:        [B, 100, T_mel_max],
        #   mel_mask:   [B, T_mel_max],}
        # 
        waves = [item[self.config.batch_wave_key] for item in batch_list]
        srs = [item[self.config.batch_sr_key] for item in batch_list]
        return {
            self.config.batch_wave_key: waves,
            self.config.batch_sr_key: srs,
        }
