from abc import ABC, abstractmethod
from pathlib import Path
from torch.utils.data import DataLoader
from config.identity_config import TrainingConfig
from trainers.base_trainer import BaseTrainer

class BaseRunner(ABC):
    def __init__(self, config: TrainingConfig, cache_dir: str, run_preprocess: bool = False):
        self.config = config
        self.cache_dir = Path(cache_dir)
        
        if run_preprocess:
            self._execute_preprocessing()

        self.dataloader = self._build_dataloader()
        self.trainer = self._build_trainer()

    @abstractmethod
    def _execute_preprocessing(self):
        pass

    @abstractmethod
    def _build_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def _build_trainer(self) -> BaseTrainer:
        pass

    @abstractmethod
    def run(self):
        pass
