from collections import OrderedDict

import torch

from utils.evaluations import assess_gene_model1d_x
from utils.datasets import LoadNumpyData, DCTDataset, NormalizedDataset, CountOccurenceDataset, CountDataset

from models.diffusion import DiffusionModel, ConditionalDiffusionModel, BinaryDiffusion
from modules.unets import UNetMLPln


def run(data_path='', result_path='', epoch_no=1, step_no=1):
    with torch.no_grad():
        # Training setup
        INPUT_DIM = 784

        LR = 1e-3

        MINI_BATCH_train = 128
        MINI_BATCH_val = 256
        NUM_STEPS = 1000

        EPOCHS = 500
        PATIENCE = 10

        if torch.cuda.is_available():
            DEVICE = 'cuda'
        else:
            DEVICE = 'cpu'
        
        print(f'Device is: {DEVICE}')
    
        # PATHS
        name = f'epoch={epoch_no}-step={step_no}.ckpt'


        # MODEL
        nnet = UNetMLPln(in_dim=INPUT_DIM, mid_dim=512, bottleneck_dim=1024, num_steps=NUM_STEPS)
        nnet = nnet.to(DEVICE)

        # -- Probabilistic Model --
        model_best = DiffusionModel(nnet=nnet, timesteps=NUM_STEPS, beta_min=1e-4, beta_max=0.02, lr=LR, sequence_length=2000, T_max=100, device=DEVICE)
        model_best = model_best.to(DEVICE)
        
        checkpoint_state_dict = torch.load(result_path + "checkpoints/" + name)
        model_dict = checkpoint_state_dict['state_dict']
        model_best.load_state_dict(model_dict)

        # DATA
        data_np = LoadNumpyData(file_name=data_path, ratios=(0.7, 0.15, 0.15))
        test_dataset_np = data_np.get_data(data_type='test')

        test_dataset = CountDataset(test_dataset_np)

        # RUN ASSESSMENT
        N = 1000
        assess_gene_model1d_x(model_best, result_path, name, test_dataset, N=N, C=1, D=1000, prior=False)


if __name__ == "__main__":
    run(data_path='data/pbmc3k_2000.npy', result_path='logging_space/VCM-Diff_data/diffusion/version_1/', epoch_no=154, step_no=2325)
