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
                 distribution="Poisson"
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
        if distribution == "Poisson":
            self.net = DeconvNet(hidden_dims=self.hidden_dims,
                                 n_celltypes=self.n_celltype,
                                 n_slices=self.n_slices,
                                 n_heads=n_heads,
                                 slice_emb_dim=slice_emb_dim,
                                 coef_fe=coef_fe,
                                 ).to(self.device)
        else: #Negative Binomial distribution
            self.net = DeconvNet_NB(hidden_dims=self.hidden_dims,
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

    def train(self, report_loss=True, step_interval=2000):
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
            
            if report_loss:
                if not step % step_interval:
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


    def cells_to_spatial(self, adata_ref_input, 
                        celltype_ref_col="celltype", # column of adata_ref_input.obs for cell type information
                        celltype_ref=None, # specify cell types to use for deconvolution
                        target_num=20, # target number of cells per spot
                        save=False, 
                        lam_sim=0.1,
                        lam_num=1e-3,
                        lam_M=1,
                        lr=2e-3, training_steps_M=20000, report_loss=True, step_interval=2000, output_path="./results"):

        import scanpy as sc

        # When map cells to spatial locations, 
        # the reference dataset needs to be processed in the same way as we used it to construct the cell-type matrix

        adata_ref = adata_ref_input.copy()
        adata_ref.var_names_make_unique()
        # Remove mt-genes
        adata_ref = adata_ref[:, np.array(~adata_ref.var.index.isna())
                              & np.array(~adata_ref.var_names.str.startswith("mt-"))
                              & np.array(~adata_ref.var_names.str.startswith("MT-"))]
        if celltype_ref is not None:
            if not isinstance(celltype_ref, list):
                raise ValueError("'celltype_ref' must be a list!")
            else:
                adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]
        else:
            celltype_counts = adata_ref.obs[celltype_ref_col].value_counts()
            celltype_ref = list(celltype_counts.index[celltype_counts > 1])
            adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]

        # Remove cells and genes with 0 counts
        sc.pp.filter_cells(adata_ref, min_genes=1)
        sc.pp.filter_genes(adata_ref, min_cells=1)

        adata_ref = adata_ref[:, self.adata_st.var.index]

        celltype_list = list(sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str))))
        if scipy.sparse.issparse(adata_ref.X):
            ref_counts = adata_ref.X.toarray()
        else:
            ref_counts = adata_ref.X

        # Generate count matrix for single cells
        ref_counts = torch.from_numpy(ref_counts).to(torch.float32).to(self.device) # N_cells x G

        celltype_onehot = np.zeros((adata_ref.shape[0], len(celltype_list)))
        for i in range(adata_ref.shape[0]):
            celltype_onehot[i, celltype_list.index(list(adata_ref.obs[celltype_ref_col].values)[i])] += 1.

        # Generate one-hot cell-type matrix for single cells
        celltype_onehot = torch.from_numpy(celltype_onehot).to(torch.float32).to(self.device) # N_cells x C

        # Generate adjusted expression matrix for spatial spots
        Y_adjusted = (torch.matmul(self.beta, self.basis) * self.lY).detach() # N_spots x G

        beta = self.beta.detach() # N_spots x C

        M = torch.zeros(adata_ref.shape[0], self.Y.shape[0]) # N_cells x N_spots
        M = M.to(self.device)
        M.requires_grad = True

        self.optimizer_M = optim.Adamax([M], lr=lr)

        for step in tqdm(range(training_steps_M)):
            M_hat = F.softmax(M, dim=1) # N_cells x N_spots

            generated_spots = torch.matmul(torch.transpose(M_hat, 0, 1), ref_counts) # N_spots x G
            loss_sim_spots = - torch.mean(F.cosine_similarity(Y_adjusted, generated_spots, dim=1))
            loss_sim_genes = - torch.mean(F.cosine_similarity(Y_adjusted, generated_spots, dim=0))

            generated_spots_prop = torch.matmul(torch.transpose(M_hat, 0, 1), celltype_onehot) # N_spots x C
            generated_spots_prop = generated_spots_prop / torch.sum(M_hat, axis=0).view(-1, 1) # Normalize generated proportions

            loss_prop = torch.mean(torch.sum((generated_spots_prop - beta) ** 2, dim=1))

            # regularizers
            target_num = adata_ref.shape[0] / self.Y.shape[0]
            reg_cell_num = torch.mean((torch.sum(M_hat, axis=0) - target_num)**2)
            reg_M = -torch.mean(M_hat * torch.log(M_hat))
            loss_M = loss_prop + lam_sim * (loss_sim_spots + loss_sim_genes) + lam_num * reg_cell_num + lam_M * reg_M
            self.optimizer_M.zero_grad()
            loss_M.backward()
            self.optimizer_M.step()
            
            if report_loss:
                if not step % step_interval:
                    print("Step: %s, Loss: %.4f, proption_loss: %.4f, spot_sim_loss: %.4f, cell_num_reg: %.4f, M_reg: %.4f" % 
                        (step, loss_M.item(), loss_prop.item(), (loss_sim_spots + loss_sim_genes).item(), reg_cell_num.item(), reg_M.item())) 

        M_hat = F.softmax(M, dim=1)
        self.M_hat = M_hat.detach().cpu().numpy()

        adata_ref.obsm['spatial_aligned'] = self.adata_st[np.argmax(self.M_hat, axis=1)].obsm['spatial_aligned']
        adata_ref.obsm['3D_coor'] = self.adata_st[np.argmax(self.M_hat, axis=1)].obsm['3D_coor']
        adata_ref.obs['slice'] = self.adata_st[np.argmax(self.M_hat, axis=1)].obs['slice'].values

        return adata_ref


