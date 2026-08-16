from pathlib import Path
import torch

from torch.utils.data import Dataset, DataLoader

from config.identity_config import TrainingConfig
from utils.processors import AudioToEmbeddingsProcessor, AudioToMelProcessor
from tqdm import tqdm


def pocket_collate_fn(batch):
    return batch[0]

class IdentityPreprocessPipeline:
    def __init__(
        self, 
        config: TrainingConfig,
        dataset: Dataset, 
        mel_processor: AudioToMelProcessor, 
        feature_extractor: AudioToEmbeddingsProcessor,
        output_dir: str | Path,
    ):
        self.mel_processor = mel_processor
        self.feature_extractor = feature_extractor
        self.output_dir = Path(output_dir)
        self.config = config

        self.mel_dir = self.output_dir / config.mel_rel_dir
        self.feats_dir = self.output_dir / config.feats_rel_dir
        
        self.mel_dir.mkdir(parents=True, exist_ok=True)
        self.feats_dir.mkdir(parents=True, exist_ok=True)

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=config.num_workers,
            collate_fn=pocket_collate_fn
        )

    def run(self):
        for batch in tqdm(self.dataloader):
            wave = batch[self.config.batch_wave_key]
            sr = batch[self.config.batch_sr_key]
            rel_path = batch[self.config.batch_path_key].with_suffix(".pt")
            
            mel_save_path = self.mel_dir / rel_path
            feats_save_path = self.feats_dir / rel_path
            
            if mel_save_path.exists() and feats_save_path.exists():
                continue
                
            mel_save_path.parent.mkdir(parents=True, exist_ok=True)
            feats_save_path.parent.mkdir(parents=True, exist_ok=True)
                
            # save target mel
            if not mel_save_path.exists():
                target_mel = self.mel_processor(wave, sr).squeeze() # [100, T_mel]
                # print(f"target shape:\t{target_mel.shape}")
                torch.save(target_mel.cpu(), mel_save_path)

            # save feats embeddings
            if not feats_save_path.exists():
                feats_outputs = self.feature_extractor(wave, sr).squeeze() # [13, T_feats, 768]
                # print(f"feats outputs shape:\t{feats_outputs.shape}")
                torch.save(feats_outputs.cpu(), feats_save_path)
            
def run_identity_preprocess(
        config: TrainingConfig,
        dataset: Dataset, 
        mel_processor: AudioToMelProcessor, 
        feature_extractor: AudioToEmbeddingsProcessor,
        output_dir: str | Path):
    run_identity_preprocess_pipeline = IdentityPreprocessPipeline(config, dataset, mel_processor, feature_extractor, output_dir)
    run_identity_preprocess_pipeline.run()