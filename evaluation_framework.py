from collections import OrderedDict

import torch

from utils.evaluations import assess_gene_model1d
from utils.datasets import LoadNumpyData, DCTDataset, NormalizedDataset, CountOccurenceDataset

from models.diffusion import DiffusionModel, ConditionalDiffusionModel, BinaryDiffusion
from modules.nnets import UNetMLP, UNetMLPbn, UNetMLPx, UNetMLPb

if __name__ == "__main__":
    with torch.no_grad():
        # PATHS
        result_dir = 'logging_space/scDiff_data/diffusion/version_2/checkpoints/'
        name = 'epoch=110-step=9546.ckpt'


        # MODEL
        checkpoint_state_dict = torch.load(result_dir + name)
        nnet_dict = checkpoint_state_dict['state_dict']

        nnet_new_dict = OrderedDict()
        for k in nnet_dict.keys():
            nnet_new_dict[k[5:]] = nnet_dict[k]
        del nnet_dict
        NUM_STEPS = 50
        # -nnet
        nnet_best = UNetMLPx(in_dim=1000, mid_dim=512, bottleneck_dim=512, num_steps=NUM_STEPS)
        nnet_best.load_state_dict(nnet_new_dict)
        # -diffusion
        model_best = ConditionalDiffusionModel(nnet=nnet_best, timesteps=NUM_STEPS, beta_min=1e-4, beta_max=0.02, lr=1e-4, sequence_length=2000, T_max=100)

        # DATA
        data_np = LoadNumpyData(file_name='data/pancreas.npy', ratios=(0.7, 0.15, 0.15))
        # train_dataset_np = data_np.get_data(data_type='train')
        test_dataset_np = data_np.get_data(data_type='test')

        # train_dataset = NormalizedDataset(train_dataset_np, normalize_counts=True)
        # test_dataset = NormalizedDataset(test_dataset_np, normalize_counts=True, scale=train_dataset.scale)
        test_dataset = CountOccurenceDataset(test_dataset_np)

        # RUN ASSESSMENT
        N = 1000
        assess_gene_model1d(model_best, result_dir, name, test_dataset, N=N, C=1, D=1000, prior=False)
