import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import multiprocessing as mp

from config.identity_config import TrainingConfig, DebugConfig
from runners.identity_runner import IdentityRunner, OnebatchIdentityRunner, OnlineIdentityRunner

import warnings
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")


def main():
    # mode = "debug"
    mode = "kaggle_training"
    # if torch.backends.mps.is_available():
    #     mp.set_start_method("forkserver", force=True)
    mp.set_start_method("fork", force=True)

    if mode == "debug":
        config = DebugConfig()
        cache_dir = "./archive/LibriTTS_R/dev-clean"
        runner = OnebatchIdentityRunner(config, cache_dir, False)
        runner.run()

    elif mode == "kaggle_training":
        config = TrainingConfig()
        cache_dir = "/kaggle/input/datasets/justsahil/libritts-r/"

        # datasets = ["dev_clean", "train_clean_100", "train_clean_360"]
        # epochs = [20, 5, 2]
        # prev_dataset = None
        # prev_epoch = None
        # checkpoint = None
        # for dataset_name, epoch in zip(datasets, epochs):
        #     dataset_dir = cache_dir + dataset_name
        #     config.epochs = epoch
        #     runner = OnlineIdentityRunner(config, dataset_dir)
        #     if prev_dataset is not None and prev_epoch is not None:
        #         checkpoint = f"checkpoint_{prev_dataset}_epoch_{prev_epoch - 1}.pt"
        #     runner.run(checkpoint=checkpoint)
        #     prev_dataset = dataset_name
        #     prev_epoch = epoch

        dataset_dir = cache_dir + "train_clean_100"
        config.epochs = 3
        runner = OnlineIdentityRunner(config, dataset_dir)
        runner.run()
    
    else:
        config = TrainingConfig()
        cache_dir = "./archive/LibriTTS_R/dev-clean"
        runner = IdentityRunner(config, cache_dir, False)
        runner.run()


if __name__ == "__main__":
    main()