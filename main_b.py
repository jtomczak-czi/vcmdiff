import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint, EarlyStopping

from models.diffusion import DiffusionModel, ColdDiffusionModel, ConditionalDiffusionModel, BinaryDiffusion
from modules.nnets import DiT, UNet1D, ResNetTime, MLPTime, UNetMLP, UNetMLPbn, UNetMLP_new, UNetMLPb
from utils.datasets import LoadNumpyData, DCTDataset, FourierDataset, NormalizedDataset, CountDataset, CountOccurenceDataset


if __name__ == '__main__':
    # Training setup
    MINI_BATCH_train = 128
    MINI_BATCH_val = 256
    NUM_STEPS = 100

    EPOCHS = 500
    PATIENCE = 10

    # Dataset
    data_np = LoadNumpyData(file_name='data/pancreas.npy', ratios=(0.7, 0.15, 0.15))
    train_dataset_np = data_np.get_data(data_type='train')
    val_dataset_np = data_np.get_data(data_type='val')
    test_dataset_np = data_np.get_data(data_type='test')

    train_dataset = CountOccurenceDataset(train_dataset_np)
    val_dataset = CountOccurenceDataset(val_dataset_np)
    test_dataset = CountOccurenceDataset(test_dataset_np)

    train_dataloader = DataLoader(train_dataset, batch_size=MINI_BATCH_train, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=MINI_BATCH_val, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=MINI_BATCH_val, shuffle=False)

    # Initialize model and trainer
    # LOGGER
    # csv_logger = CSVLogger("/mnt/czi-sci-ai/intrinsic-variation-gene-ex/project_gene_regulation/scDiff_data", name="diffusion.py")
    csv_logger = CSVLogger("logging_space/scDiff_data", name="diffusion")

    # MODEL
    # -- Parameterization --
    nnet = UNetMLPb(in_dim=1000, mid_dim=512, bottleneck_dim=512, num_steps=NUM_STEPS)

    # -- Probabilistic Model --
    model = BinaryDiffusion(nnet=nnet, timesteps=NUM_STEPS, beta_min=1e-4, beta_max=0.02, lr=1e-4, sequence_length=2000, T_max=100)

    # TRAINER
    # -checkpoints
    progress_bar_callback = TQDMProgressBar(refresh_rate=50, leave=True)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=PATIENCE, verbose=False, mode="min")
    lr_callback = LearningRateMonitor(logging_interval="step")
    model_checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=5)
    # -trainer
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="cpu", devices=1, logger=csv_logger, callbacks=[lr_callback, early_stop_callback, progress_bar_callback, model_checkpoint_callback], log_every_n_steps=40)

    # Train model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Test model
    trainer.test(model, dataloaders=test_dataloader)
