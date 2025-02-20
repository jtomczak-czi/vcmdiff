import numpy as np

import matplotlib.pyplot as plt

import torch
import torch_dct as dct

import anndata
import scanpy as sc
import umap

from utils.mmd import MMDLoss, BrayCurtisKernel, TanimotoKernel
from utils.simple_functions import log1p


def assess_gene_model1d_b(model_best, result_dir, name, test_data, N: int=1000, C: int=1, D: int=1000, prior=True):
    b_data = test_data.gene_occurence[:N, :, :].squeeze()

    x1 = torch.round(torch.clamp(b_data, 0.))

    # get a synthetic sample
    gen_data = model_best.sample(shape=(b_data.shape[0], 1, b_data.shape[1])).squeeze()
    gen_data = torch.gt(gen_data, 0.).float()

    x2 = torch.round(torch.clamp(gen_data, 0., 1.))

    # Calculate the MMD loss
    mmd = MMDLoss(kernel=TanimotoKernel())
    print(f'MMD: {mmd(x1, x2).item()}')
    # Caclulate the absolute diff between means
    print(f'ABS diff: {torch.abs(x1.mean(0) - x2.mean(0)).mean().item()}')
    f = open(result_dir + name + '_evaluation.txt', "w")
    f.write(f'MMD: {mmd(x1, x2).item()}\nABS diff: {torch.abs(x1.mean(0) - x2.mean(0)).mean().item()}')
    f.close()

    # HISTOGRAMS
    # Define the bins
    bins = np.linspace(0, 300, 50)
    plt.hist(x1.mean(0), alpha=0.75, bins=bins, label='Data-mean')
    plt.hist(x2.mean(0).detach(), alpha=0.5, bins=bins, label='Samples-mean')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_hist_mean.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    bins = np.linspace(0, 300, 50)
    plt.hist(x1.std(0), alpha=0.75, bins=bins, label='Data-std')
    plt.hist(x2.std(0).detach(), alpha=0.5, bins=bins, label='Samples-std')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_hist_std.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('Hist samples - DONE!')

    # BARS
    # Define the bins
    plt.bar(range(x1.shape[1]), x1.mean(0), alpha=0.75, label='Data-mean')
    plt.bar(range(x1.shape[1]), x2.mean(0).detach(), alpha=0.5, label='Samples-mean')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_bar_mean.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    plt.bar(range(x1.shape[1]), x1.std(0), alpha=0.75, label='Data-std')
    plt.bar(range(x1.shape[1]), x2.std(0).detach(), alpha=0.5, label='Samples-std')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_bar_std.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    plt.bar(range(x1.shape[1]), abs(x1.mean(0) - x2.mean(0).detach()), alpha=1., label='Difference')
    plt.title(f'Avg. abs difference: {torch.abs(x1.mean(0) - x2.mean(0).detach()).mean()}')
    plt.legend()
    plt.savefig(result_dir + name + '_bar_difference.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('Hist samples - DONE!')

    # UMAP
    adata_true = anndata.AnnData(x1.cpu().detach().numpy())
    adata_generated = anndata.AnnData(x2.cpu().detach().numpy())

    adata_full = anndata.concat(
        [adata_true, adata_generated],
        keys=["true", "generated"],
        label="data",
    )

    sc.pp.normalize_total(adata_full)
    sc.pp.log1p(adata_full)
    sc.pp.pca(adata_full)
    sc.pp.neighbors(adata_full)
    sc.tl.umap(adata_full)

    sc.pl.umap(adata_full, color=['data'], save='_' + name, alpha=0.5)


def assess_gene_model1d(model_best, result_dir, name, test_data, N: int=1000, C: int=1, D: int=1000, prior=True):
    if prior is True:
        total_counts_real = torch.randint(low=1600, high=1601, size=(N,1))
        total_counts_gen = torch.randint(low=1600, high=1601, size=(N, 1))

    # get a real sample
    # scales = test_data.scale.squeeze(1)

    # data = test_data.data[:N, :, :].squeeze()
    # data = (0.5 * data + 0.5) * scales
    # if prior is True:
    #     data = data * total_counts_real
    #
    # x1 = torch.round(torch.clamp(data, 0.))
    # print(x1.sum(1))
    #
    # # get a synthetic sample
    # gen_data = model_best.sample(shape=(N, C, D)).squeeze()
    # gen_data = torch.clamp(gen_data, -1., 1.)
    # gen_data = (0.5 * gen_data + 0.5) * scales
    # if prior is True:
    #     gen_data = gen_data * total_counts_gen
    #
    # x2 = torch.round(torch.clamp(gen_data, 0.))
    # print(x2.sum(1))

    x_data = test_data.gene_count[:N, :, :].squeeze()
    b_data = test_data.gene_occurence[:N, :, :].squeeze()

    x1 = torch.round(torch.clamp(x_data, 0.))

    # get a synthetic sample
    gen_data = model_best.sample(shape=(x_data.shape[0], 1, x_data.shape[1]), b=b_data.unsqueeze(1)).squeeze()

    x2 = torch.round(torch.clamp(gen_data, 0.))

    # Calculate the MMD loss
    mmd = MMDLoss(kernel=BrayCurtisKernel())
    print(f'MMD: {mmd(x1, x2).item()}')
    # Caclulate the absolute diff between means
    print(f'ABS diff: {torch.abs(x1.mean(0) - x2.mean(0)).mean().item()}')
    f = open(result_dir + name + '_evaluation.txt', "w")
    f.write(f'MMD: {mmd(x1, x2).item()}\nABS diff: {torch.abs(x1.mean(0) - x2.mean(0)).mean().item()}')
    f.close()

    # HISTOGRAMS
    # Define the bins
    bins = np.linspace(0, 300, 50)
    plt.hist(x1.mean(0), alpha=0.75, bins=bins, label='Data-mean')
    plt.hist(x2.mean(0).detach(), alpha=0.5, bins=bins, label='Samples-mean')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_hist_mean.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    bins = np.linspace(0, 300, 50)
    plt.hist(x1.std(0), alpha=0.75, bins=bins, label='Data-std')
    plt.hist(x2.std(0).detach(), alpha=0.5, bins=bins, label='Samples-std')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_hist_std.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('Hist samples - DONE!')

    # BARS
    # Define the bins
    plt.bar(range(x1.shape[1]), x1.mean(0), alpha=0.75, label='Data-mean')
    plt.bar(range(x1.shape[1]), x2.mean(0).detach(), alpha=0.5, label='Samples-mean')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_bar_mean.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    plt.bar(range(x1.shape[1]), x1.std(0), alpha=0.75, label='Data-std')
    plt.bar(range(x1.shape[1]), x2.std(0).detach(), alpha=0.5, label='Samples-std')
    plt.title(name)
    plt.legend()
    plt.savefig(result_dir + name + '_bar_std.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    plt.bar(range(x1.shape[1]), abs(x1.mean(0) - x2.mean(0).detach()), alpha=1., label='Difference')
    plt.title(f'Avg. abs difference: {torch.abs(x1.mean(0) - x2.mean(0).detach()).mean()}')
    plt.legend()
    plt.savefig(result_dir + name + '_bar_difference.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print('Hist samples - DONE!')

    # UMAP
    adata_true = anndata.AnnData(x1.cpu().detach().numpy())
    adata_generated = anndata.AnnData(x2.cpu().detach().numpy())

    adata_full = anndata.concat(
        [adata_true, adata_generated],
        keys=["true", "generated"],
        label="data",
    )

    sc.pp.normalize_total(adata_full)
    sc.pp.log1p(adata_full)
    sc.pp.pca(adata_full)
    sc.pp.neighbors(adata_full)
    sc.tl.umap(adata_full)

    sc.pl.umap(adata_full, color=['data'], save='_' + name, alpha=0.5)


    # xx1 = log1p(x1)
    # xx2 = log1p(x2)
    #
    # reducer = umap.UMAP()
    # emd1 = reducer.fit_transform(xx1)
    # emb2 = reducer.fit_transform(xx2)






