import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
import os
from STitch3D.networks import *


class Model():

    def __init__(self, adata_st, adata_basis,
                 hidden_dims=[512, 128],
                 n_heads=1,
                 slice_emb_dim=16,
                 coef_fe=0.1,
                 training_steps=20000,
                 lr=2e-3,
                 seed=1234,
                 ):

        self.training_steps = training_steps

        self.adata_st = adata_st
        self.celltypes = list(adata_basis.obs.index)

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.hidden_dims = [adata_st.shape[1]] + hidden_dims
        self.n_celltype = adata_basis.shape[0]
        self.n_slices = len(sorted(set(adata_st.obs["slice"].values)))

        # build model
        self.net = DeconvNet(hidden_dims=self.hidden_dims,
                             n_celltypes=self.n_celltype,
                             n_slices=self.n_slices,
                             n_heads=n_heads,
                             slice_emb_dim=slice_emb_dim,
                             coef_fe=coef_fe,
                             ).to(self.device)

        self.optimizer = optim.Adamax(list(self.net.parameters()), lr=lr)

        # read data
        if scipy.sparse.issparse(adata_st.X):
            self.X = torch.from_numpy(adata_st.X.toarray()).float().to(self.device)
        else:
            self.X = torch.from_numpy(adata_st.X).float().to(self.device)
        self.A = torch.from_numpy(np.array(adata_st.obsm["graph"])).float().to(self.device)
        self.Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(self.device)
        self.lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
        self.slice = torch.from_numpy(np.array(adata_st.obs["slice"].values)).long().to(self.device)
        self.basis = torch.from_numpy(np.array(adata_basis.X)).float().to(self.device)

    def train(self):
        self.net.train()
        for step in tqdm(range(self.training_steps)):
            loss = self.net(adj_matrix=self.A,
                            node_feats=self.X,
                            count_matrix=self.Y,
                            library_size=self.lY,
                            slice_label=self.slice,
                            basis=self.basis)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            if not step % 200:
                print("Step: %s, Loss: %.4f, d_loss: %.4f, f_loss: %.4f" % (step, loss.item(), self.net.decon_loss.item(), self.net.features_loss.item()))  


    def eval(self, adata_st_list_raw, save=False, output_path="./results"):
        self.net.eval()
        self.Z, self.beta, self.alpha, self.gamma = self.net.evaluate(self.A, self.X, self.slice)

        if save == True:
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        # add learned representations to full ST adata object
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names, ].values
        if save == True:
            cell_reps.to_csv(os.path.join(output_path, "representation.csv"))

        # add deconvolution results to original anndata objects
        b = self.beta.detach().cpu().numpy()
        n_spots = 0
        adata_st_decon_list = []
        for i, adata_st_i in enumerate(adata_st_list_raw):
            adata_st_i.obs.index = adata_st_i.obs.index + "-slice%d" % i
            decon_res = pd.DataFrame(b[n_spots:(n_spots+adata_st_i.shape[0]), :], 
                                     columns=self.celltypes)
            decon_res.index = adata_st_i.obs.index
            adata_st_i.obs = adata_st_i.obs.join(decon_res)
            n_spots += adata_st_i.shape[0]
            adata_st_decon_list.append(adata_st_i)

            if save == True:
                decon_res.to_csv(os.path.join(output_path, "prop_slice%d.csv" % i))
                adata_st_i.write(os.path.join(output_path, "res_adata_slice%d.h5ad" % i))

        # Save 3d coordinates
        if save == True:
            coor_3d = pd.DataFrame(data=self.adata_st.obsm['3D_coor'], index=self.adata_st.obs.index, columns=['x', 'y', 'z'])
            coor_3d.to_csv(os.path.join(output_path, "3D_coordinates.csv"))

        return adata_st_decon_list

