import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt
from STitch3D.align_tools import *
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from matplotlib import cm


def align_spots(adata_st_list_input, # list of spatial transcriptomics datasets
                method="icp", # "icp" or "paste"
                data_type="Visium", # a spot has six nearest neighborhoods if "Visium", four nearest neighborhoods otherwise
                coor_key="spatial", # "spatial" for visium; key for the spatial coordinates used for alignment
                tol=0.01, # parameter for "icp" method; tolerance level
                test_all_angles=False, # parameter for "icp" method; whether to test multiple rotation angles or not
                plot=False,
                paste_alpha=0.1,
                paste_dissimilarity="kl"
                ):
    # Align coordinates of spatial transcriptomics

    # The first adata in the list is used as a reference for alignment
    adata_st_list = adata_st_list_input.copy()

    if plot:
        # Choose colors
        cmap = cm.get_cmap('rainbow', len(adata_st_list))
        colors_list = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(len(adata_st_list))]

        # Plot spots before alignment
        plt.figure(figsize=(5, 5))
        plt.title("Before alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm[coor_key][:, 0], 
                adata_st_list[i].obsm[coor_key][:, 1], 
                c=colors_list[i],
                label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list)//13 + 1))
        plt.show()


    if (method == "icp") or (method == "ICP"):
        print("Using the Iterative Closest Point algorithm for alignemnt.")
        # Detect edges
        print("Detecting edges...")
        point_cloud_list = []
        for adata in adata_st_list:
            # Use in-tissue spots only
            if 'in_tissue' in adata.obs.columns:
                adata = adata[adata.obs['in_tissue'] == 1]
            if data_type == "Visium":
                loc_x = adata.obs.loc[:, ["array_row"]]
                loc_x = np.array(loc_x) * np.sqrt(3)
                loc_y = adata.obs.loc[:, ["array_col"]]
                loc_y = np.array(loc_y)
                loc = np.concatenate((loc_x, loc_y), axis=1)
                pairwise_loc_distsq = np.sum((loc.reshape([1,-1,2]) - loc.reshape([-1,1,2])) ** 2, axis=2)
                n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
            else:
                loc_x = adata.obs.loc[:, ["array_row"]]
                loc_x = np.array(loc_x)
                loc_y = adata.obs.loc[:, ["array_col"]]
                loc_y = np.array(loc_y)
                loc = np.concatenate((loc_x, loc_y), axis=1)
                pairwise_loc_distsq = np.sum((loc.reshape([1,-1,2]) - loc.reshape([-1,1,2])) ** 2, axis=2)
                min_distsq = np.sort(np.unique(pairwise_loc_distsq), axis=None)[1]
                n_neighbors = np.sum(pairwise_loc_distsq < (min_distsq * 3), axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 7)).astype(np.float32)
            point_cloud_list.append(adata.obsm[coor_key][edge == 1].copy())

        # Align edges
        print("Aligning edges...")
        trans_list = []
        adata_st_list[0].obsm["spatial_aligned"] = adata_st_list[0].obsm[coor_key].copy()
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            if test_all_angles == True:
                for angle in [0., np.pi * 1 / 3, np.pi * 2 / 3, np.pi, np.pi * 4 / 3, np.pi * 5 / 3]:
                    R = np.array([[np.cos(angle), np.sin(angle), 0], 
                                  [-np.sin(angle), np.cos(angle), 0], 
                                  [0, 0, 1]]).T
                    T, distances, _ = icp(transform(point_cloud_list[i+1], R), point_cloud_list[i], tolerance=tol)
                    if angle == 0:
                        loss_best = np.mean(distances)
                        angle_best = angle
                        R_best = R
                        T_best = T
                    else:
                        if np.mean(distances) < loss_best:
                            loss_best = np.mean(distances)
                            angle_best = angle
                            R_best = R
                            T_best = T
                T = T_best @ R_best
            else:
                T, _, _ = icp(point_cloud_list[i+1], point_cloud_list[i], tolerance=tol)
            trans_list.append(T)
        # Tranform
        for i in range(len(adata_st_list) - 1):
            point_cloud_align = adata_st_list[i+1].obsm[coor_key].copy()
            for T in trans_list[:(i+1)][::-1]:
                point_cloud_align = transform(point_cloud_align, T)
            adata_st_list[i+1].obsm["spatial_aligned"] = point_cloud_align

    elif (method == "paste") or (method == "PASTE"):
        print("Using PASTE algorithm for alignemnt.")
        # Align spots
        print("Aligning spots...")
        pis = []
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            pi = pairwise_align_paste(adata_st_list[i], adata_st_list[i+1], coor_key=coor_key,
                                      alpha = paste_alpha, dissimilarity = paste_dissimilarity)
            pis.append(pi)
        # Tranform
        S1, S2  = generalized_procrustes_analysis(adata_st_list[0].obsm[coor_key], 
                                                  adata_st_list[1].obsm[coor_key], 
                                                  pis[0])
        adata_st_list[0].obsm["spatial_aligned"] = S1
        adata_st_list[1].obsm["spatial_aligned"] = S2
        for i in range(1, len(adata_st_list) - 1):
            S1, S2 = generalized_procrustes_analysis(adata_st_list[i].obsm["spatial_aligned"], 
                                                     adata_st_list[i+1].obsm[coor_key], 
                                                     pis[i])
            adata_st_list[i+1].obsm["spatial_aligned"] = S2

    if plot:
        plt.figure(figsize=(5, 5))
        plt.title("After alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm["spatial_aligned"][:, 0], 
                adata_st_list[i].obsm["spatial_aligned"][:, 1], 
                c=colors_list[i],
                label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list)//13 + 1))
        plt.show()

    return adata_st_list


def preprocess(adata_st_list_input, # list of spatial transcriptomics (ST) anndata objects
               adata_ref_input, # reference single-cell anndata object
               celltype_ref_col="celltype", # column of adata_ref_input.obs for cell type information
               sample_col=None, # column of adata_ref_input.obs for batch labels
               celltype_ref=None, # specify cell types to use for deconvolution
               n_hvg_group=500, # number of highly variable genes for reference anndata
               three_dim_coor=None, # if not None, use existing 3d coordinates in shape [# of total spots, 3]
               coor_key="spatial_aligned", # "spatial_aligned" by default
               rad_cutoff=None, # cutoff radius of spots for building graph
               rad_coef=1.1, # if rad_cutoff=None, rad_cutoff is the minimum distance between spots multiplies rad_coef
               slice_dist_micron=None, # pairwise distances in micrometer for reconstructing z-axis 
               prune_graph_cos=False, # prune graph connections according to cosine similarity
               cos_threshold=0.5, # threshold for pruning graph connections
               c2c_dist=100, # center to center distance between nearest spots in micrometer
               ):

    adata_st_list = adata_st_list_input.copy()

    print("Finding highly variable genes...")
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

    # Concatenate ST adatas
    for i in range(len(adata_st_list)):
        adata_st_new = adata_st_list[i].copy()
        adata_st_new.var_names_make_unique()
        # Remove mt-genes
        adata_st_new = adata_st_new[:, (np.array(~adata_st_new.var.index.str.startswith("mt-")) 
                                    & np.array(~adata_st_new.var.index.str.startswith("MT-")))]
        adata_st_new.obs.index = adata_st_new.obs.index + "-slice%d" % i
        adata_st_new.obs['slice'] = i
        if i == 0:
            adata_st = adata_st_new
        else:
            genes_shared = adata_st.var.index & adata_st_new.var.index
            adata_st = adata_st[:, genes_shared].concatenate(adata_st_new[:, genes_shared], index_unique=None)

    adata_st.obs["slice"] = adata_st.obs["slice"].values.astype(int)

    # Take gene intersection
    genes = list(adata_st.var.index & adata_ref.var.index)
    adata_ref = adata_ref[:, genes]
    adata_st = adata_st[:, genes]

    # Select hvgs
    adata_ref_log = adata_ref.copy()
    sc.pp.log1p(adata_ref_log)
    hvgs = select_hvgs(adata_ref_log, celltype_ref_col=celltype_ref_col, num_per_group=n_hvg_group)

    print("%d highly variable genes selected." % len(hvgs))
    adata_ref = adata_ref[:, hvgs]

    print("Calculate basis for deconvolution...")
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.normalize_total(adata_ref, target_sum=1)
    celltype_list = list(sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str))))

    basis = np.zeros((len(celltype_list), len(adata_ref.var.index)))
    if sample_col is not None:
        sample_list = list(sorted(set(adata_ref.obs[sample_col].values.astype(str))))
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp_list = []
            for j in range(len(sample_list)):
                s = sample_list[j]
                tmp = adata_ref[(adata_ref.obs[celltype_ref_col].values.astype(str) == c) & 
                                (adata_ref.obs[sample_col].values.astype(str) == s), :].X
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                if tmp.shape[0] >= 3:
                    tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
            tmp_mean = np.mean(tmp_list, axis=0)
            if scipy.sparse.issparse(tmp_mean):
                tmp_mean = tmp_mean.toarray()
            print("%d batches are used for computing the basis vector of cell type <%s>." % (len(tmp_list), c))
            basis[i, :] = tmp_mean
    else:
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp = adata_ref[adata_ref.obs[celltype_ref_col].values.astype(str) == c, :].X
            if scipy.sparse.issparse(tmp):
                tmp = tmp.toarray()
            basis[i, :] = np.mean(tmp, axis=0).reshape((-1))

    adata_basis = ad.AnnData(X=basis)
    df_gene = pd.DataFrame({"gene": adata_ref.var.index})
    df_gene = df_gene.set_index("gene")
    df_celltype = pd.DataFrame({"celltype": celltype_list})
    df_celltype = df_celltype.set_index("celltype")
    adata_basis.obs = df_celltype
    adata_basis.var = df_gene
    adata_basis = adata_basis[~np.isnan(adata_basis.X[:, 0])]

    print("Preprocess ST data...")
    # Store counts and library sizes for Poisson modeling
    st_mtx = adata_st[:, hvgs].X.copy()
    if scipy.sparse.issparse(st_mtx):
        st_mtx = st_mtx.toarray()
    adata_st.obsm["count"] = st_mtx
    st_library_size = np.sum(st_mtx, axis=1)
    adata_st.obs["library_size"] = st_library_size

    # Normalize ST data
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    adata_st = adata_st[:, hvgs]
    if scipy.sparse.issparse(adata_st.X):
        adata_st.X = adata_st.X.toarray()

    # Build a graph for spots across multiple slices
    print("Start building a graph...")

    # Build 3D coordinates 
    if three_dim_coor is None:

        # The first adata in adata_list is used as a reference for computing cutoff radius of spots
        adata_st_ref = adata_st_list[0].copy()
        loc_ref = np.array(adata_st_ref.obsm[coor_key])
        pair_dist_ref = pairwise_distances(loc_ref)
        min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]

        if rad_cutoff is None:
            # The radius is computed base on the attribute "adata.obsm['spatial']"
            rad_cutoff = min_dist_ref * rad_coef
        print("Radius for graph connection is %.4f." % rad_cutoff)

        # Use the attribute "adata.obsm['spatial_aligned']" to build a global graph
        if slice_dist_micron is None:
            loc_xy = pd.DataFrame(adata_st.obsm['spatial_aligned']).values
            loc_z = np.zeros(adata_st.shape[0])
            loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)
        else:
            if len(slice_dist_micron) != (len(adata_st_list) - 1):
                raise ValueError("The length of 'slice_dist_micron' should be the number of adatas - 1 !")
            else:
                loc_xy = pd.DataFrame(adata_st.obsm['spatial_aligned']).values
                loc_z = np.zeros(adata_st.shape[0])
                dim = 0
                for i in range(len(slice_dist_micron)):
                    dim += adata_st_list[i].shape[0]
                    loc_z[dim:] += slice_dist_micron[i] * (min_dist_ref / c2c_dist)
                loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)

    # If 3D coordinates already exists
    else:
        if rad_cutoff is None:
            raise ValueError("Please specify 'rad_cutoff' for finding 3D neighbors!")
        loc = three_dim_coor

    pair_dist = pairwise_distances(loc)
    G = (pair_dist < rad_cutoff).astype(float)

    if prune_graph_cos:
        pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine") # 1 - cosine_similarity
        G_cos = (pair_dist_cos < (1 - cos_threshold)).astype(float)
        G = G * G_cos

    print('%.4f neighbors per cell on average.' % (np.mean(np.sum(G, axis=1)) - 1))
    adata_st.obsm["graph"] = G
    adata_st.obsm["3D_coor"] = loc

    return adata_st, adata_basis


def select_hvgs(adata_ref, celltype_ref_col, num_per_group=200):
    sc.tl.rank_genes_groups(adata_ref, groupby=celltype_ref_col, method="t-test", key_added="ttest", use_raw=False)
    markers_df = pd.DataFrame(adata_ref.uns['ttest']['names']).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes


def calculate_impubasis(adata_st_input, #st anndata object (should be one of the output from STitch3D.utils.preprocess)
                        adata_ref_input, # reference single-cell anndata object (raw data)
                        celltype_ref_col="celltype", # column of adata_ref_input.obs for cell type information
                        sample_col=None, # column of adata_ref_input.obs for batch labels
                        celltype_ref=None, # specify cell types to use for deconvolution
                        ):
    
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

    # Calculate single cell library sizes
    hvgs = adata_st_input.var.index
    adata_ref_ls = adata_ref[:, hvgs]
    sc.pp.filter_cells(adata_ref_ls, min_genes=1)
    adata_ref = adata_ref[adata_ref_ls.obs.index, :]
    # ref_ls: library size (only account for hvgs) of single cells in ref
    if scipy.sparse.issparse(adata_ref_ls.X):
        ref_ls = np.sum(adata_ref_ls.X.toarray(), axis=1).reshape((-1,1))
        adata_ref.obsm["forimpu"] = adata_ref.X.toarray() / ref_ls
    else:
        ref_ls = np.sum(adata_ref_ls.X, axis=1).reshape((-1,1))
        adata_ref.obsm["forimpu"] = adata_ref.X / ref_ls

    # Calculate basis for imputation
    celltype_list = list(sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str))))
    basis_impu = np.zeros((len(celltype_list), len(adata_ref.var.index)))
    if sample_col is not None:
        sample_list = list(sorted(set(adata_ref.obs[sample_col].values.astype(str))))
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp_list = []
            for j in range(len(sample_list)):
                s = sample_list[j]
                tmp = adata_ref[(adata_ref.obs[celltype_ref_col].values.astype(str) == c) & 
                                (adata_ref.obs[sample_col].values.astype(str) == s), :].obsm["forimpu"]
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                if tmp.shape[0] >= 3:
                    tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
            tmp_mean = np.mean(tmp_list, axis=0)
            if scipy.sparse.issparse(tmp_mean):
                tmp_mean = tmp_mean.toarray()
            print("%d batches are used for computing the basis vector of cell type <%s>." % (len(tmp_list), c))
            basis_impu[i, :] = tmp_mean
    else:
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp = adata_ref[adata_ref.obs[celltype_ref_col].values.astype(str) == c, :].obsm["forimpu"]
            if scipy.sparse.issparse(tmp):
                tmp = tmp.toarray()
            basis_impu[i, :] = np.mean(tmp, axis=0).reshape((-1))

    adata_basis_impu = ad.AnnData(X=basis_impu)
    df_gene = pd.DataFrame({"gene": adata_ref.var.index})
    df_gene = df_gene.set_index("gene")
    df_celltype = pd.DataFrame({"celltype": celltype_list})
    df_celltype = df_celltype.set_index("celltype")
    adata_basis_impu.obs = df_celltype
    adata_basis_impu.var = df_gene
    adata_basis_impu = adata_basis_impu[~np.isnan(adata_basis_impu.X[:, 0])]
    return adata_basis_impu


