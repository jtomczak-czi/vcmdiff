import numpy as np

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint, EarlyStopping

from models.diffusion import DiffusionModel, ColdDiffusionModel, ConditionalDiffusionModel
from modules.nnets import DiT, UNet1D, ResNetTime, MLPTime, UNetMLP, UNetMLPbn, UNetMLP_new, UNetMLPx
from utils.datasets import LoadNumpyData, DCTDataset, FourierDataset, NormalizedDataset, CountDataset, CountOccurenceDataset


def run():
    # Training setup
    MINI_BATCH_train = 128
    MINI_BATCH_val = 256
    NUM_STEPS = 100

    EPOCHS = 500
    PATIENCE = 10

    # Dataset
    # dataset = FourierDataset()
    # dataset = DCTDataset()
    data_np = LoadNumpyData(file_name='data/pancreas.npy', ratios=(0.7, 0.15, 0.15))
    train_dataset_np = data_np.get_data(data_type='train')
    val_dataset_np = data_np.get_data(data_type='val')
    test_dataset_np = data_np.get_data(data_type='test')

    # train_dataset = DCTDataset(train_dataset_np)
    # val_dataset = DCTDataset(val_dataset_np, shift=train_dataset.shift, scale=train_dataset.scale)
    # test_dataset = DCTDataset(test_dataset_np, shift=train_dataset.shift, scale=train_dataset.scale)
    # train_dataset = NormalizedDataset(train_dataset_np, normalize_counts=False)
    # val_dataset = NormalizedDataset(val_dataset_np, normalize_counts=False, shift=train_dataset.shift, scale=train_dataset.scale)
    # test_dataset = NormalizedDataset(test_dataset_np, normalize_counts=False, shift=train_dataset.shift, scale=train_dataset.scale)
    # train_dataset = CountDataset(train_dataset_np)
    # val_dataset = CountDataset(val_dataset_np)
    # test_dataset = CountDataset(test_dataset_np)
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
    # in_channels=1
    # hidden_size=256
    # depth=6
    # num_heads=8
    # sequence_length = 1000

    # nnet = DiT(sequence_length, in_channels=in_channels, hidden_size=hidden_size, depth=depth, num_heads=num_heads)
    # nnet = UNet1D(in_channels=in_channels, base_channels=16, embed_dim=32, bottleneck_dim=64)
    # nnet = ResNetTime(in_features=1000, mid_features=64, out_features=32, num_blocks=4, num_steps=NUM_STEPS)
    # nnet = MLPTime(in_features=1000, mid_features=256, num_blocks=3, num_steps=NUM_STEPS)
    # nnet = UNetMLP(in_dim=1000, mid_dim=256, bottleneck_dim=256, num_steps=NUM_STEPS)
    nnet = UNetMLPx(in_dim=1000, mid_dim=512, bottleneck_dim=512, num_steps=NUM_STEPS)

    # -- Probabilistic Model --
    # model = DiffusionModel(nnet=nnet, timesteps=NUM_STEPS, beta_min=1e-4, beta_max=0.02, lr=1e-4, sequence_length=2000, T_max=100)
    # model = ColdDiffusionModel(nnet, timesteps=NUM_STEPS, lr=1e-4, sequence_length=2000, T_max=100)
    model = ConditionalDiffusionModel(nnet=nnet, timesteps=NUM_STEPS, beta_min=1e-4, beta_max=0.02, lr=1e-4, sequence_length=2000, T_max=100)

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


if __name__ == '__main__':
    run()

    print('-----===== Training is done =====-----')
