# This file is part of the Cell_free_secrecy distribution
# Copyright 2025 b<>com. All rights reserved. 

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#%%
import h5py
import numpy as np
from numpy import linalg as LA
from sklearn.manifold import Isomap, LocallyLinearEmbedding,SpectralEmbedding,TSNE
from sklearn.neighbors import NearestNeighbors
import umap
import torch
from torch import Tensor
from torch.utils.data import Dataset
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray
n_l = 4
n_L = 4
M = n_l*n_L                         ## number of AP (ALICE)
L = 6                   ## number of antennas per AP
n_r = 1                      ## number of antennas per U

# Load the dataset and separate training and test datasets
path_name = "./data_base_channel/" 
dataset_filename = path_name + "NLOS_channelpositiondataset_M_3072.npz"
data = np.load(file=dataset_filename)
channels = torch.tensor(data["channels"], dtype=torch.complex64)
positions = torch.tensor(data["positions"], dtype=torch.complex64).real.to(torch.float64)

#nonzero_mask = torch.any(channels != 0, dim=1)  # [N]

# Filter both h and p
#h_cleaned = channels[nonzero_mask]
#p_cleaned = positions[nonzero_mask]

# Shuffle indices
N = channels.shape[0]
seed = 4
torch.manual_seed(seed)
indices = torch.randperm(N)

# Split
train_idx = indices[:int(0.6 * N)]
test_idx = indices[int(0.6 * N):]

# Create splits
train_channels = channels[train_idx]
train_positions = positions[train_idx].real
test_channels = channels[test_idx]
test_positions = positions[test_idx].real
#%%
def MR_precoder0(H,P):
    return np.sqrt(P)*(H.conj().T)/(LA.norm(H,'fro'))

def ZF_precoder0(H,P,Kb):
    Hbob  = H[:Kb,:]
    H_pinv = LA.pinv(Hbob)
    norms = LA.norm(H_pinv,'fro')
    scaling = np.sqrt(P)/norms               
    return H_pinv *scaling

def pcZF0(H,P,Kb,Ke):
    Hbob  = H[:Kb,:]
    Heve = H[-Ke:,:]
    H_pinv = LA.pinv(Heve)
    P_e = np.eye(M*L,dtype=np.complex64) - H_pinv @ Heve
    Hbob_proj = Hbob @ P_e
    return ZF_precoder0(Hbob_proj,P,Kb)

# Kruskal stress:
def kruskal_stress(dist_X:Tensor, dist_X_embedded:Tensor):
    beta = torch.sum(dist_X*dist_X_embedded)/torch.sum(dist_X_embedded**2)
    stress = torch.sqrt(torch.sum((dist_X - beta*dist_X_embedded)**2)/torch.sum(dist_X**2))
    return stress

# Trustworthiness :
    # les voisins dans la nouvelle représentation sont-ils proches dans la
    # représentation d'origine ? Oui --> 1, Non --> 0
def trustworthiness(dist_X:Tensor, dist_X_embedded:Tensor, n_neighbors:int=5):
    device=dist_X.device
    dist_X = dist_X.detach().clone()
    dist_X_embedded = dist_X_embedded.detach().clone()

    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    dist_X.fill_diagonal_(float('Inf'))
    ind_X = torch.argsort(dist_X, dim=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    dist_X_embedded.fill_diagonal_(float('Inf'))
    ind_X_embedded = torch.topk(dist_X_embedded, n_neighbors, largest=False, sorted=False).indices
    

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    n_samples = dist_X.shape[0]
    inverted_index = torch.zeros((n_samples, n_samples), dtype=int).to(device)
    ordered_indices = torch.arange(n_samples + 1).to(device)
    inverted_index[ordered_indices[:-1, None],
                   ind_X] = ordered_indices[1:]
    ranks = inverted_index[ordered_indices[:-1, None],
                           ind_X_embedded] - n_neighbors
    t = torch.sum(ranks[ranks > 0], dtype=torch.int64)
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t

# Continuity :
    # les voisins dans la représentation d'origine sont-ils proches dans la
    # nouvelle représentation ? Oui --> 1, Non --> 0
def continuity(dist_X, dist_X_embedded, n_neighbors=5):
    c = trustworthiness(dist_X_embedded, dist_X, n_neighbors)
    return c

def optimal_SRR(locs,z):
    """Compute the optimal Scaling, Rotation and/or Reflection matrices.
    Reference: https://arxiv.org/pdf/2005.12242.pdf (Section V.B.).

    Parameters
    ----------
    locs : (N,2) array
        Original locations.
    z : (N,2) array
        Channel chart.

    Returns
    -------
    (2,2) array
        Optimal SRR matrix.
    """
    locs_mean = torch.mean(locs,dim=0)
    z_mean = torch.mean(z,dim=0)
    
    locs_std = torch.std(locs,dim=0)
    z_std = torch.std(z,dim=0)

    locs_hat = locs-locs_mean
    z_hat = (z-z_mean)*(locs_std/z_std)

    U,S,V_H = LA.svd(z_hat.T@locs_hat)

    return U@V_H

def compute_CT_TW(locs,pseudo_locs,title,bool_save,fn_save):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not isinstance(locs,torch.Tensor):
        locs = torch.tensor(locs,device=device)
    if not isinstance(pseudo_locs,torch.Tensor):
        pseudo_locs = torch.tensor(pseudo_locs,device=device)

    dist_locs = torch.cdist(locs,locs)
    dist_embed = torch.cdist(pseudo_locs,pseudo_locs)

    size_emb_base = dist_embed.shape[0]

    tw_charting = np.empty(10)
    ct_charting = np.empty(10)

    k_base = (np.linspace(1,10,10)/100)*size_emb_base

    idx = 0
    for k in k_base:
        tw_charting[idx] = trustworthiness(dist_locs,dist_embed,int(k)).item()
        ct_charting[idx] = continuity(dist_locs,dist_embed,int(k)).item()
        idx += 1

    fig, ax = plt.subplots()
    ax.plot(np.arange(1,11),tw_charting,label='TW')
    ax.plot(np.arange(1,11),ct_charting,label='CT')
    ax.set_xlabel(r'Number of neighbours (\% of charting locations nb)')
    ax.set_ylabel(r'TW/CT')
    ax.set_title(title)
    ax.legend()

    idx = (k_base == (5/100)*size_emb_base)
    print(f'{title}: 5%: CT {ct_charting[idx][0]:.3f}, TW: {tw_charting[idx][0]:.3f}')

    if bool_save:
        fig.savefig(fn_save+'TW_CT.pdf',dpi=800,bbox_inches='tight')

def phase_incent_dictance(u:torch.Tensor,v:torch.Tensor=None):
    norm_u = torch.linalg.vector_norm(x=u,ord=2,dim=-1)
    if v is None:
        v = u
        norm_v = norm_u
    else:
        norm_v = torch.linalg.vector_norm(x=v,ord=2,dim=-1)

    uHv = torch.vdot(u,v).real

    return torch.sqrt(torch.maximum(torch.tensor(0),2-2*uHv/(norm_u*norm_v)))

def phase_insent_dictance_vec(X: torch.Tensor):
    """
    Vectorized version of the phase-insent distance for all pairs in X.
    Input:
        X: tensor of shape [N, D] (can be complex)
    Output:
        D: tensor of shape [N, N], where D[i, j] = distance(X[i], X[j])
    """
    # Normalize all vectors
    norms = torch.linalg.vector_norm(X, ord=2, dim=1, keepdim=True)  # [N, 1]
    X_norm = X / norms  # [N, D]

    # Compute Hermitian inner product (vdot) between all pairs: X Xᴴ
    inner = torch.matmul(X_norm, X_norm.conj().T)  # [N, N], complex

    # Real part only (vdot returns complex), clamp to avoid negatives
    
    real_part = torch.clamp(inner.real, min=-1.0, max=1.0)

    # Compute distance
    D = torch.sqrt(torch.clamp(2 - 2 * real_part, min=0.0))  # [N, N]
    return D

def phase_insent_distance_single(X: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
    """
    Compute phase-insent distances between a single new channel and all channels in X.

    Args:
        X: [N, D] complex tensor (dataset of channels)
        h_new: [D] complex tensor (new channel)

    Returns:
        distances: [N] tensor of distances
    """
    # Normalize existing dataset
    norms_X = torch.linalg.vector_norm(X, ord=2, dim=1)  # [N]
    X_norm = X / norms_X.unsqueeze(1)                   # [N, D]

    # Normalize new channel
    norm_h = torch.linalg.vector_norm(h_new, ord=2)     # scalar
    h_norm = h_new / norm_h                             # [D]

    # Hermitian inner product
    inner = torch.matmul(X_norm, h_norm.conj())         # [N]
    real_part = torch.clamp(inner.real, min=-1.0, max=1.0)

    # Distance
    return torch.sqrt(torch.clamp(2 - 2 * real_part, min=0.0))

def phase_insent_distance_batch(X: torch.Tensor, H_new: torch.Tensor) -> torch.Tensor:
    """
    Compute phase-incent distances between a batch of new channels and all channels in X.

    Args:
        X: [N, D] complex tensor (dataset of channels)
        H_new: [B, D] complex tensor (batch of new channels)

    Returns:
        distances: [B, N] tensor of distances
    """
    # Normalize dataset
    norms_X = torch.linalg.vector_norm(X, ord=2, dim=1)   # [N]
    X_norm = X / norms_X.unsqueeze(1)                     # [N, D]

    # Normalize new batch
    norms_H = torch.linalg.vector_norm(H_new, ord=2, dim=1)  # [B]
    H_norm = H_new / norms_H.unsqueeze(1)                    # [B, D]

    # Hermitian inner product between H_norm and X_norm
    inner = torch.matmul(H_norm, X_norm.conj().T)         # [B, N]
    real_part = torch.clamp(inner.real, min=-1.0, max=1.0)

    # Distances
    return torch.sqrt(torch.clamp(2 - 2 * real_part, min=0.0))

def chart_init(chans: torch.Tensor, num_ngbr: int, dim_chart: int = 2, method: str = "umap"):
    """
    Computes a low-dimensional chart of the channels using different manifold learning methods.
    
    Parameters:
        chans      : torch.Tensor of shape (n_samples, n_features)
        num_ngbr   : int, number of neighbors for the manifold method
        dim_chart  : int, target embedding dimension
        method     : str, one of ["isomap", "umap", "lle", "laplacian", "tsne"]
        
    Returns:
        torch.Tensor of shape (n_samples, dim_chart)
    """

    # Compute distance matrix
    Dist = phase_insent_dictance_vec(chans).detach().cpu().numpy()

    # --- Isomap ---
    if method == "isomap":
        model = Isomap(n_neighbors=num_ngbr, n_components=dim_chart, metric="precomputed")
        Z = model.fit_transform(Dist)
        return torch.from_numpy(Z).float()

    # --- UMAP ---
    elif method == "umap":
        model = umap.UMAP(n_neighbors=num_ngbr, n_components=dim_chart, metric="precomputed",random_state=seed)
        Z = model.fit_transform(Dist)
        return torch.from_numpy(Z).float()

    # --- Locally Linear Embedding (LLE) ---
    elif method == "lle":
        # Build kNN graph from precomputed distances
        nbrs = NearestNeighbors(n_neighbors=num_ngbr, metric="precomputed").fit(Dist)
        kneighbors_graph = nbrs.kneighbors_graph(Dist,n_neighbors=num_ngbr, mode="connectivity")

        model = LocallyLinearEmbedding(
            n_neighbors=num_ngbr,
            n_components=dim_chart,
            method="standard",
            eigen_solver="auto"
        )

        # LLE requires a dense matrix
        Z = model.fit_transform(kneighbors_graph.toarray())
        return torch.from_numpy(Z).float()

    # --- Laplacian / Spectral Embedding ---
    elif method == "laplacian":
        # Build kNN connectivity graph from precomputed distances
        nbrs = NearestNeighbors(n_neighbors=num_ngbr, metric="precomputed").fit(Dist)
        kneighbors_graph = nbrs.kneighbors_graph(Dist,n_neighbors=num_ngbr, mode="connectivity")

        model = SpectralEmbedding(
            n_components=dim_chart,
            n_neighbors=num_ngbr,
            affinity="precomputed_nearest_neighbors"
        )

        # Sparse connectivity graph is fine
        Z = model.fit_transform(kneighbors_graph)
        return torch.from_numpy(Z).float()

    # --- t-SNE ---
    elif method == "tsne":
        tsne_dim = min(dim_chart, 3)  # t-SNE max 3 dimensions
        model = TSNE(n_components=tsne_dim, perplexity=max(num_ngbr,5), metric="precomputed", init="random",random_state=seed)
        Z = model.fit_transform(Dist)
        return torch.from_numpy(Z).float()

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: isomap, umap, lle, laplacian, tsne")
    
def HTk(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Hard thresholding: keep k largest values, zero the rest.
    """
    if k >= x.numel():
        return x
    topk_vals, topk_idx = torch.topk(x, k)
    out = torch.zeros_like(x)
    out[topk_idx] = topk_vals
    return out

def similarity_subsampling(D,Z,chans_charting,locs_charting,N_tilde):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Subsample for D_tilde, Z_tilde
    D_tilde = D[:,:N_tilde].detach().clone()
    Z_tilde = Z[:N_tilde,:].detach().clone()

    D_c = D[:,N_tilde:].detach().clone()
    Z_c = Z[N_tilde:,:].detach().clone()

    chans_charting_tilde = chans_charting.detach().clone()[:N_tilde,:]
    locs_charting = locs_charting.detach().clone()
    locs_charting_tilde = locs_charting[:N_tilde,:]
    locs_charting_c = locs_charting[N_tilde:,:]

    for i in tqdm(range(Z_c.shape[0])):
        h_i = D_c[:,i]
        z_i = Z_c[i,:]
        locs_i = locs_charting_c[i,:]

        # Find the max similarity in D_tilde
        similarities = torch.abs(torch.matmul(Z_tilde,torch.conj(z_i)))/ (torch.sqrt(torch.sum(torch.abs(Z_tilde)**2,dim=1)) * torch.sqrt(torch.sum(torch.abs(z_i)**2)))
        max_sim_with_new = torch.max(similarities)

        N = Z_tilde.shape[0]
        norm_Z_tilde = torch.sqrt(torch.sum(torch.abs(Z_tilde)**2,dim=1))
        similarity_matrix = torch.abs(torch.matmul(Z_tilde, torch.conj(Z_tilde).T))/(norm_Z_tilde.unsqueeze(1) * norm_Z_tilde.unsqueeze(0))

        upper_triangle_indices = torch.triu_indices(N, N, offset=1, device=device)
        upper_triangle_similarities = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]

        max_sim, max_sim_idx = torch.max(upper_triangle_similarities, dim=0)
        best_pair = (upper_triangle_indices[0, max_sim_idx].item(), upper_triangle_indices[1, max_sim_idx].item())

        if max_sim_with_new < max_sim:
            # Replace h_m or h_n with h_i
            r = best_pair[0] if np.random.rand() < 0.5 else best_pair[1]

            D_tilde[:,r] = h_i
            Z_tilde[r, :] = z_i

            chans_charting_tilde[r,:] = h_i
            locs_charting_tilde[r,:] = locs_i

    chans_charting = chans_charting_tilde
    locs_charting = locs_charting_tilde

    return D_tilde, Z_tilde, chans_charting, locs_charting

def similarity_subsampling1(D,Z,N_tilde):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Subsample for D_tilde, Z_tilde
    D_tilde = D[:,:N_tilde].clone()
    Z_tilde = Z[:N_tilde,:].clone()

    D_c = D[:,N_tilde:].clone()
    Z_c = Z[N_tilde:,:].clone()

    for i in tqdm(range(Z_c.shape[0])):
        h_i = D_c[:,i]
        z_i = Z_c[i,:]

        # Find the max similarity in D_tilde
        similarities = torch.abs(torch.matmul(Z_tilde,torch.conj(z_i)))/ (torch.sqrt(torch.sum(torch.abs(Z_tilde)**2,dim=1)) * torch.sqrt(torch.sum(torch.abs(z_i)**2)))
        max_sim_with_new = torch.max(similarities)

        N = Z_tilde.shape[0]
        norm_Z_tilde = torch.sqrt(torch.sum(torch.abs(Z_tilde)**2,dim=1))
        similarity_matrix = torch.abs(torch.matmul(Z_tilde, torch.conj(Z_tilde).T))/(norm_Z_tilde.unsqueeze(1) * norm_Z_tilde.unsqueeze(0))

        upper_triangle_indices = torch.triu_indices(N, N, offset=1, device=device)
        upper_triangle_similarities = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]

        max_sim, max_sim_idx = torch.max(upper_triangle_similarities, dim=0)
        best_pair = (upper_triangle_indices[0, max_sim_idx].item(), upper_triangle_indices[1, max_sim_idx].item())

        if max_sim_with_new < max_sim:
            # Replace h_m or h_n with h_i
            r = best_pair[0] if np.random.rand() < 0.5 else best_pair[1]

            D_tilde[:,r] = h_i
            Z_tilde[r, :] = z_i

    return D_tilde, Z_tilde

def similarity_subsampling2(D,chans_charting,locs_charting,N_tilde):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Subsample for D_tilde, Z_tilde
    D_tilde = D[:,:N_tilde].detach().clone()

    D_c = D[:,N_tilde:].detach().clone()
    chans_charting_tilde = chans_charting.detach().clone()[:N_tilde,:]

    locs_charting = locs_charting.detach().clone()
    locs_charting_tilde = locs_charting[:N_tilde,:]
    locs_charting_c = locs_charting[N_tilde:,:]

    for i in tqdm(range(D_c.shape[1])):
        h_i = D_c[:,i]
        locs_i = locs_charting_c[i,:]

        # Find the max similarity in D_tilde
        similarities = phase_insent_distance_single(D_tilde.T,h_i)
        max_sim_with_new = torch.min(similarities)

        N = D_tilde.shape[1]
        similarity_matrix = phase_insent_dictance_vec(D_tilde.T)

        upper_triangle_indices = torch.triu_indices(N, N, offset=1, device=device)
        upper_triangle_similarities = similarity_matrix[upper_triangle_indices[0], upper_triangle_indices[1]]

        max_sim, max_sim_idx = torch.min(upper_triangle_similarities, dim=0)
        best_pair = (upper_triangle_indices[0, max_sim_idx].item(), upper_triangle_indices[1, max_sim_idx].item())

        if max_sim_with_new > max_sim:
            # Replace h_m or h_n with h_i
            r = best_pair[0] if np.random.rand() < 0.5 else best_pair[1]

            D_tilde[:,r] = h_i

            chans_charting_tilde[r,:] = h_i
            locs_charting_tilde[r,:] = locs_i

    chans_charting = chans_charting_tilde
    locs_charting = locs_charting_tilde
    return D_tilde, chans_charting, locs_charting

def encoder(D: torch.Tensor, Z: torch.Tensor, h: torch.Tensor, k: int) -> torch.Tensor:
    """
    Encode channel vector h using dictionary D and initial chart Z.
    
    Args:
        D (torch.Tensor): [N, M] complex-valued dictionary matrix.
        Z (torch.Tensor): [N, d] real-valued chart positions.
        h (torch.Tensor): [M] complex-valued channel vector.
        k (int): number of neighbors to keep in hard thresholding.

    Returns:
        z (torch.Tensor): [d] embedded location of h in the chart.
    """
    # normalize dictionary and query
    D_norm = D / torch.linalg.vector_norm(D, dim=1, keepdim=True)
    h_norm = h / torch.linalg.vector_norm(h)

    # 1: Correlation (row-wise Hermitian inner products)
    a = torch.sum(D_norm.conj() * h_norm, dim=1)  # [N]

    # 2: Modulus
    b = torch.abs(a)  # [N]

    # 3: Hard thresholding
    c = HTk(b, k)  # [N]

    # 4: L1 Normalization
    norm = torch.sum(c)
    d = c / norm if norm > 0 else c

    # 5: Weighted sum over chart coordinates
    z = torch.matmul(d, Z)  # [d]

    return z

def k_nn_insent_phase(X: torch.Tensor, h: torch.Tensor, k: int) :

    # Compute distance vector
    dist = phase_insent_distance_single(X,h)
    values, indices = torch.topk(dist, k, largest=False)
    return indices

def get_k_nearest_neighbors(chart: torch.Tensor, location: torch.Tensor, k: int):
    """
    Returns the indices of the k nearest neighbors to a given location in the chart.

    Args:
        chart (torch.Tensor): Tensor of shape [N, D], where N is the number of points and D is the chart dimension.
        location (torch.Tensor): Tensor of shape [D] or [1, D], the query point in the same chart space.
        k (int): Number of nearest neighbors to return.

    Returns:
        torch.Tensor: Tensor of shape [k] containing the indices of the k nearest neighbors.
    """
    if location.dim() == 1:
        location = location.unsqueeze(0)  # [1, D]

    # Compute Euclidean distances to all points
    dists = torch.norm(chart - location, dim=1)  # [N]

    # Get indices of the k smallest distances
    knn_indices = torch.topk(dists, k=k, largest=False).indices  # [k]

    return knn_indices

def color_by_proximity(X):
    # Normalize to [0, 1] for consistent colormap use
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    # Map 2D positions to colors using a colormap
    # Use x and y as hue + brightness (or just use 1D if needed)
    cmap = cm.get_cmap('viridis')
    color_vals = X_norm[:, 0] + 0.5 * X_norm[:, 1]   # weighted blend
    color_vals = (color_vals - color_vals.min()) / (color_vals.max() - color_vals.min())
    colors = cmap(color_vals)
    return colors

def assign_grid_colors(X, n_bins_x=4, n_bins_y=4, colormap_name='tab20'):
    """
    Assign colors to each point in X by binning into a grid.
    Returns colors and the bin edges info needed to assign new points.
    """
    # Calculate min/max for normalization
    X_min = np.min(X.numpy(),axis=0)
    X_max =  np.max(X.numpy(),axis=0)

    # Normalize points to [0,1]
    X_norm = (X - X_min) / (X_max - X_min)

    # Bin indices for each dimension
    bin_x = np.floor(X_norm[:, 0] * n_bins_x)
    bin_y = np.floor(X_norm[:, 1] * n_bins_y)
    
    # Clip edges
    bin_x = np.clip(bin_x, 0, n_bins_x - 1)
    bin_y = np.clip(bin_y, 0, n_bins_y - 1)

    bin_ids = bin_y * n_bins_x + bin_x
    n_bins = n_bins_x * n_bins_y
    bin_ids = bin_ids.numpy().astype(int)

    cmap = plt.get_cmap(colormap_name)
    cmap_colors = cmap(np.linspace(0, 1, n_bins))

    colors = cmap_colors[bin_ids]

    # Return colors and binning parameters needed for new points
    binning_info = {
        'X_min': X_min,
        'X_max': X_max,
        'n_bins_x': n_bins_x,
        'n_bins_y': n_bins_y,
        'cmap_colors': cmap_colors
    }
    return colors, binning_info

def get_color_for_new_point(x_new, binning_info):
    """
    Given a new point x_new, assign the bin color based on binning_info.
    """
    X_min = binning_info['X_min']
    X_max = binning_info['X_max']
    n_bins_x = binning_info['n_bins_x']
    n_bins_y = binning_info['n_bins_y']
    cmap_colors = binning_info['cmap_colors']

    # Normalize new point
    x_norm = (x_new - X_min) / (X_max - X_min)

    # Find bin indices
    bin_x = int(np.clip(np.floor(x_norm[0] * n_bins_x), 0, n_bins_x - 1))
    bin_y = int(np.clip(np.floor(x_norm[1] * n_bins_y), 0, n_bins_y - 1))

    bin_id = bin_y * n_bins_x + bin_x

    return cmap_colors[bin_id]

""" def generate_training_scene(scene_name,knn_mth,eve_idx):
    xe = test_positions[eve_idx]
    if knn_mth == "benchmark":
        knn_He, knn_pos = get_eves_ref(eve_ref, K_e-1, subcarrier)
    elif knn_mth == "CC":
        knn_He, knn_pos = get_eves_CC(eve_ref, K_e-1, subcarrier, method=mtd,num_neib_ml=5,dim_ls=3, N_tilde= N_tilde_values[-1])
    elif knn_mth == "dist-insent":
        knn_He, knn_pos = get_eves_dist_insent(eve_ref,K_e-1,subcarrier)
    scene = load_scene(scene_name)
    scene.tx_array = PlanarArray(num_cols=n_l, 
                              num_rows=n_L, 
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="iso",
                              polarization="V")
    scene.rx_array = PlanarArray(num_cols=n_r, 
                              num_rows=n_r, 
                              vertical_spacing=0.5,
                              horizontal_spacing=0.5,
                              pattern="iso",
                              polarization="V")

    # Create transmitters and add in the scene
    for i, pos in enumerate(tx_pos):
        tx = Transmitter(name = f"tx{i}", position = pos, orientation = [0,np.pi/2,0])
        scene.add(tx)
    mask = torch.ones(train_channels.size(0), dtype=torch.bool)
    mask[knn_pos] = False
    for i, pos in enumerate(train_positions[mask]):
        rx = Receiver(name = f"rx{i}", position = (pos[0].item(),pos[1].item(),1.5),color=(0.0,0.0,0.0))
        scene.add(rx)
    for i,pos in enumerate(train_positions[knn_pos]):
        rx = Receiver(name = f"rx-{i}", position = (pos[0].item(),pos[1].item(),1.5),color=(0.0,1.0,0.0))
        scene.add(rx)
    rx = Receiver(name='Eve',position = (xe[0].item(),xe[1].item(),1.5),color=(1.0,0.0,0.0))
    scene.add(rx)
    return scene """

def plot_results(results, dims, *args):
    if len(args) == 0:
        # --- Version simple ---
        plt.figure(figsize=(12, 6))

        # Trustworthiness
        plt.subplot(1, 2, 1)
        for name in results:
            plt.plot(dims, results[name]["trust"], marker="o", label=f"{name}")
        plt.title("Trustworthiness vs. Chart Dimension")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Trustworthiness 5%")
        plt.legend()

        # Continuity
        plt.subplot(1, 2, 2)
        for name in results:
            plt.plot(dims, results[name]["cont"], marker="s", label=f"{name}")
        plt.title("Continuity vs. Chart Dimension")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Continuity 5%")
        plt.legend()

        plt.tight_layout()
        plt.show()

    elif len(args) == 1:
        # --- Version avec neighbors ---
        neighbors = args[0]
        for mthd in results:
            plt.figure(figsize=(10, 5))

            # Trustworthiness
            plt.subplot(1, 2, 1)
            for num_ngbr in neighbors:
                plt.plot(dims, results[mthd]["trust"][num_ngbr], label=f"ngbr={num_ngbr}")
            plt.title(f"Trustworthiness - {mthd}")
            plt.xlabel("Embedding dimension")
            plt.ylabel("Trustworthiness 5%")
            plt.legend()

            # Continuity
            plt.subplot(1, 2, 2)
            for num_ngbr in neighbors:
                plt.plot(dims, results[mthd]["cont"][num_ngbr], label=f"ngbr={num_ngbr}")
            plt.title(f"Continuity - {mthd}")
            plt.xlabel("Embedding dimension")
            plt.ylabel("Continuity 5%")
            plt.legend()

            plt.tight_layout()
            plt.show()

    else:
        raise TypeError("plot_results attend 2 ou 3 arguments (results, dims[, neighbors])")

def compute_coverage_map(hfreq, precoder, subcarrier, batch_size=500):
    """
    Compute coverage map in dB for a given subcarrier and precoder.

    Args:
        hfreq: Channel frequency responses (N x L x M x S).
        precoder: Precoder matrix (M*L x Kb).
        subcarrier: Index of the subcarrier to evaluate.
        batch_size: Number of samples per batch.

    Returns:
        Coverage map in dB (N,).
    """
    num_samples = hfreq.shape[0]
    ML, Kb = precoder.shape

    cm_list = []

    for batch_idx in tqdm(range(int(np.ceil(num_samples / batch_size)))):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        # Flatten channel for this batch: (batch_size, L, M) → (batch_size, M*L)
        htemp = np.reshape(
            hfreq[start:end, :, :, subcarrier],
            (end - start, ML)
        )

        # Effective channel after precoding: (batch_size, Kb)
        g_eff = htemp @ precoder

        # Received power per sample: sum over Kb streams
        g_power = np.sum(np.abs(g_eff) ** 2, axis=1)

        cm_list.append(g_power)

    cm = np.concatenate(cm_list, axis=0)
    return 10 * np.log10(cm)

def compute_coverage_map_vectorized(hfreq, precoder, subcarrier):
    """
    Vectorized computation of coverage map in dB for a given subcarrier and precoder.
    
    Args:
        hfreq: Channel frequency responses (N x L x M x S).
        precoder: Precoder matrix (M*L x Kb).
        subcarrier: Index of the subcarrier to evaluate.

    Returns:
        Coverage map in dB (N,).
    """
    N, L, M, S = hfreq.shape
    ML, Kb = precoder.shape

    # Flatten channel: (N, L, M) → (N, M*L)
    h_flat = hfreq[:, :, :, subcarrier].reshape(N, ML)

    # Effective channel: (N, Kb)
    g_eff = h_flat @ precoder

    # Received power per sample: sum over Kb streams
    g_power = np.sum(np.abs(g_eff) ** 2, axis=1)

    # Convert to dB
    return 10 * np.log10(g_power)

def knn_channels(chans_X ,knn, subcarr):
    k = np.size(knn)
    chans = chans_X[knn]  # pick the channels for the k nearest neighbors
    hfreq = chans.reshape((k, num_subcarriers, M*L))  # reshape to (k, subcarriers, antennas)
    hfreq = hfreq.permute(0, 2, 1)  # swap axes to (k, antennas, subcarriers)
    hfreq = hfreq.detach().cpu().numpy()
    return hfreq[:, :, subcarr]  # return only selected subcarrier

def get_eves_ref(eve_ref_idx: int, num_neib_eve: int,subcarrier: int):
    knn = get_k_nearest_neighbors(train_positions,test_positions[eve_ref_idx],num_neib_eve).detach().cpu().numpy()
    H = knn_channels(train_channels,knn,subcarrier)
    pos = train_positions[knn].detach().cpu().numpy()
    return H, pos

def get_eves_CC(eve_ref_idx: int, num_neib_eve: int, subcarrier: int, method: str = "umap", num_neib_ml: int =5, dim_ls:int =5, N_tilde: int=100):
    computed_chart = chart_init(train_channels,num_neib_ml,dim_ls,method)
    reduced_train_channels, reduced_computed_chart, redduced_train_channels, reduced_train_positions = similarity_subsampling(train_channels.T,computed_chart,train_channels,train_positions,N_tilde)
    ze = encoder(reduced_train_channels.T,reduced_computed_chart,test_channels[eve_ref_idx],k=25)
    knn = get_k_nearest_neighbors(reduced_computed_chart,ze,num_neib_eve).detach().cpu().numpy()
    H = knn_channels(reduced_train_channels.T,knn,subcarrier)
    pos = reduced_train_positions[knn].detach().cpu().numpy()
    return H, pos

def get_eves_dist_insent(eve_ref_idx: int, num_neib_eve: int,subcarrier: int, N_tilde: int=100):
    reduced_D, reduced_train_channels, reduced_train_positions  = similarity_subsampling2(train_channels.T,train_channels,train_positions,N_tilde)
    knn = k_nn_insent_phase(reduced_D.T,test_channels[eve_ref_idx],num_neib_eve).detach().cpu().numpy()
    H = knn_channels(reduced_D.T ,knn, subcarrier)
    pos = reduced_train_positions[knn].detach().cpu().numpy()
    return H, pos

def inverse_distance_weight(di, beta=2, eps=1e-6):
    """
    Compute inverse distance weights.

    Parameters
    ----------
    di : array-like
        Vector of distances (N,).
    beta : float
        Exponent for inverse distance weighting.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    w : np.ndarray
        Vector of weights (N,).
    """
    di = np.asarray(di)
    return 1.0 / (di + eps)**beta

def exponential_weight(di, lam=10.0):
    di = np.asarray(di)
    return np.exp(-di/lam)

def gaussian_weight(di, sigma=5.0):
    di = np.asarray(di)
    return np.exp(-0.5*(di/sigma)**2)
    
def dist_weighted_sum_chan_gain(cm: np.ndarray, eve_ref_idx: int, nb_points: int, weight_mth: str = "average"):
    voisins = get_k_nearest_neighbors(train_positions,test_positions[eve_ref_idx],nb_points).detach().cpu().numpy()
    gains_lin = np.power(10,cm[voisins]/10)
    dist = np.linalg.norm(train_positions[voisins] - test_positions[eve_ref_idx],axis=1)

    # Choose weight method
    if weight_mth == "inverse": 
        f_di = inverse_distance_weight(dist)
    elif weight_mth == "exp":
        f_di = exponential_weight(dist)
    elif weight_mth == "average":
        f_di = np.ones(np.size(voisins),np.float64)
    elif weight_mth == "gaussian":
        f_di = gaussian_weight(dist)
    else:
        raise ValueError(f"Unknown weight method '{weight_mth}'. Choose from: inverse, exp, average, gaussian")

    S_bar = np.sum(f_di*gains_lin)/np.sum(f_di)
    return 10*np.log10(S_bar)

def compute_rate_eve(cm: np.ndarray, eve_ref_idx: int, nb_points: int, weight_mth: str, sigma2: np.ndarray, subcarrier ):
    """ Computes spectral efficiencies for given invsigma2"""
    eff_gain = dist_weighted_sum_chan_gain(cm, eve_ref_idx, nb_points, weight_mth)
    eff_gain_lin = np.power(10,eff_gain/10)
    sinr = np.zeros(np.size(sigma2),np.float64)
    for i,sigma in enumerate(sigma2):
        sinr[i] = (eff_gain_lin)/(sigma)
    return np.log2(1+sinr)

def compute_sum_rate_eve(cm: np.ndarray,
                         eve_ref_idx: int,
                         nb_points: int,
                         sigma2: np.float64,
                         num_subcarriers: int,
                         eve_precoder: np.ndarray,
                         subcarrier: int):
    """Compute Eve's spectral efficiency for different noise variances sigma2."""
    
    # 1. Nearest neighbors
    voisins = get_k_nearest_neighbors(train_positions,
                                      test_positions[eve_ref_idx],
                                      nb_points).detach().cpu().numpy()
    
    # 2. Initialize effective gains
    eff_gain_lin = np.zeros(nb_points + 1, np.float64)
    eff_gain_lin[:nb_points] = np.power(10, cm[voisins] / 10)
    
    # 3. Compute coverage map for the Eve reference
    eve_freq = test_channels[eve_ref_idx].reshape(1, num_subcarriers, L, M) \
                                          .permute(0, 2, 3, 1) \
                                          .detach().cpu().numpy()
    
    eff_gain_scalar = compute_coverage_map_vectorized(eve_freq, eve_precoder, subcarrier).item()
    eff_gain_lin[-1] = np.power(10, eff_gain_scalar/10)
    
    se = np.mean(np.log2(1 + (eff_gain_lin / sigma2)))
    return se

def compute_sum_rate_bob(H, W, sigma2):
    """Compute Bob's sum rate given channel H, precoder W, and noise variance sigma2."""
    G = H @ W  # effective channel (Kb x Kb)

    eff_gain = np.square(np.abs(np.diag(G)))  # desired signal per Bob
    interf_gain = np.sum(np.square(np.abs(G - np.diag(np.diag(G)))), axis=1)  # interference power per Bob

    sinr = eff_gain / (interf_gain + sigma2)
    sr = np.sum(np.log2(1 + sinr))
    return sr

def compute_sigma2_lin(H,snr,P):
    kb = H.shape[0]
    G = H @ ZF_precoder0(H,P,kb)
    eff_gain = np.square(np.abs(np.diag(G)))
    #snr_lin = 10**(-snr/10)
    snr_lin_inv = np.power(10,-snr/10)
    sigma2_k_lin = np.zeros((kb,np.size(snr)))
    for i in range(kb):
        sigma2_k_lin[i] = eff_gain[i]*snr_lin_inv
    return sigma2_k_lin

# Rotate functions
def rotate_points_90ccw(points):
    x, y = points[:,0], points[:,1]
    return np.stack([-y, x], axis=1)

def rotate_point_90ccw(point):
    x, y = point[0], point[1]
    return np.array([-y, x])
#%% ====================================================================
# NLDR Hyperparameters Tuning
# ====================================================================
# This section performs hyperparameter tuning for various Non-Linear 
# Dimensionality Reduction (NLDR) methods (Isomap, UMAP, t-SNE) by 
# evaluating trustworthiness and continuity scores over a range of 
# dimensions and neighborhood sizes.
# ====================================================================

# Define NLDR methods and hyperparameter ranges
methods = ["isomap", "umap", "tsne"]  # NLDR algorithms
dims = range(2, 11)                    # Target embedding dimensions
neighbors = range(4, 16)               # Neighborhood sizes for NLDR

# Initialize results dictionary to store trustworthiness and continuity
results = {
    method: {"trust": {}, "cont": {}} for method in methods
}

# Loop over each method, number of neighbors, and embedding dimension
for method in methods:
    for num_ngbr in neighbors:
        results[method]["trust"][num_ngbr] = []
        results[method]["cont"][num_ngbr] = []
        for dim in dims:
            print(f"Fitting {method} at dim={dim}, neighbors={num_ngbr}")
            
            # Generate NLDR embedding using chart_init
            chart = chart_init(
                chans=train_channels,
                num_ngbr=num_ngbr,
                dim_chart=dim,
                method=method
            )

            # Determine number of neighbors for trustworthiness/continuity
            nb_TWCT = int((1 / 100) * chart.shape[0])

            # Compute pairwise distances in original space and embedding
            dist_locs = torch.cdist(train_positions, train_positions)
            dist_embed = torch.cdist(chart, chart)

            # Compute trustworthiness and continuity
            trust = trustworthiness(dist_locs, dist_embed, n_neighbors=nb_TWCT)
            cont = continuity(dist_locs, dist_embed, n_neighbors=nb_TWCT)

            # Store results
            results[method]["trust"][num_ngbr].append(trust)
            results[method]["cont"][num_ngbr].append(cont)

# Convert results into numpy arrays for easier storage and analysis
dims_len = len(dims)
trust_arr = np.zeros((len(methods), len(neighbors), dims_len))
cont_arr = np.zeros((len(methods), len(neighbors), dims_len))

for m_idx, method in enumerate(methods):
    for n_idx, num_ngbr in enumerate(neighbors):
        trust_arr[m_idx, n_idx, :] = results[method]["trust"][num_ngbr]
        cont_arr[m_idx, n_idx, :] = results[method]["cont"][num_ngbr]

# Save results to HDF5 file for later analysis
filename = f"results_TWCT1_NLOS_N_{train_channels.shape[0]}_D_{train_channels.shape[1]}.h5"
with h5py.File(filename, "w") as f:
    f.create_dataset("trust", data=trust_arr)
    f.create_dataset("continuity", data=cont_arr)
    f.create_dataset("methods", data=np.string_(methods))
    f.create_dataset("neighbors", data=neighbors)
#%% ====================================================================
# SE-Eve Computation for Various NLDR and Benchmark Methods
# ====================================================================
# This section evaluates the eavesdropper's exclusion region average spectral
# efficiency (SE_eve) using different NLDR-based methods, distance-insensitive 
# neighbor selection, and conventional benchmarks. It iterates over SNR values 
# and computes SE for each method.
# ====================================================================

# -----------------------------
# Hyperparameters for NLDR methods
# -----------------------------
max_neib_isomap = 5
max_dim_isomap = 9

max_neib_umap = 6
max_dim_umap = 10

max_neib_tsne = 14

# System parameters
num_subcarriers = 32
subcarrier = 16
Pt = 1  # Transmit power
fractions = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]  # Fraction of training set
N_tilde_values = [int(train_channels.shape[0] * f) for f in fractions]

# -----------------------------
# Bob and Eve positions
# -----------------------------
bob = [5311, 5440]  # Bob indices in the training dataset
K_b = len(bob)      # Number of Bobs
eve_ref = 784       # Eve index in the test dataset
eve_xy = test_positions[eve_ref]  # Eve position (x, y)

# -----------------------------
# Channel preparation
# -----------------------------
recovered_transposed = train_channels.reshape(train_channels.shape[0], num_subcarriers, L, M)
h_freq_recovered = recovered_transposed.permute(0, 2, 3, 1)
h_freq = h_freq_recovered.detach().cpu().numpy()

# Bob's channel matrix for selected subcarrier
H_b = np.reshape(h_freq[bob, :, :, subcarrier], (K_b, M * L))

# Noise variance for each SNR
snr = np.arange(-5, 36, 1)
sigma2_lin = np.mean(compute_sigma2_lin(H_b, snr, Pt), axis=0)

# -----------------------------
# Methods to evaluate
# -----------------------------
CM_methods = ["benchmark", "dist-insent", "isomap", "umap", "tsne"]
num_maps = len(CM_methods)
K_e = 33  # Number of eavesdroppers (including Eve)
SE_eve = np.zeros((num_maps + 1, len(snr)))  # +1 for conventional ZF

# Reference neighbor channels for Eve
knn_ref, knn_pos_ref = get_eves_ref(eve_ref, K_e - 1, subcarrier)
H_e_ref = np.concatenate(
    (np.reshape(test_channels[eve_ref, subcarrier * M * L:(subcarrier + 1) * M * L], (1, M * L)), 
     knn_ref),
    axis=0
)

# -----------------------------
# Loop over each method
# -----------------------------
for i, method_name in enumerate(CM_methods):
    # --- Neighbor selection based on method ---
    if method_name == "benchmark":
        knn_He, knn_pos = get_eves_ref(eve_ref, K_e - 1, subcarrier)
    elif method_name == "dist-insent":
        knn_He, knn_pos = get_eves_dist_insent(eve_ref, K_e - 1, subcarrier, N_tilde=N_tilde_values[-1])
    else:
        # Charting-based NLDR methods
        if method_name == "isomap":
            knn_He, knn_pos = get_eves_CC(
                eve_ref, K_e - 1, subcarrier, method=method_name, 
                num_neib_ml=max_neib_isomap, dim_ls=max_dim_isomap, 
                N_tilde=N_tilde_values[-1]
            )
        elif method_name == "umap":
            knn_He, knn_pos = get_eves_CC(
                eve_ref, K_e - 1, subcarrier, method=method_name, 
                num_neib_ml=max_neib_umap, dim_ls=max_dim_umap, 
                N_tilde=N_tilde_values[-1]
            )
        elif method_name == "tsne":
            knn_He, knn_pos = get_eves_CC(
                eve_ref, K_e - 1, subcarrier, method=method_name, 
                num_neib_ml=max_neib_tsne, dim_ls=3, 
                N_tilde=N_tilde_values[-1]
            )
    
    # Concatenate Eve's reference channel with neighbors
    H_e = np.concatenate(
        (np.reshape(test_channels[eve_ref, subcarrier * M * L:(subcarrier + 1) * M * L], (1, M * L)), 
         knn_He),
        axis=0
    )

    # Complete channel matrix including Bob and Eve
    Hmatrix = np.concatenate((H_b, H_e), axis=0)

    # Compute coverage map using linear precoding (pcZF0)
    covmap = compute_coverage_map_vectorized(h_freq, pcZF0(Hmatrix, Pt, K_b, K_e), subcarrier)

    # Compute SE for each SNR
    for sig_idx, sigma in enumerate(sigma2_lin):
        SE_eve[i, sig_idx] = compute_sum_rate_eve(
            covmap, eve_ref, K_e - 1, sigma, num_subcarriers,
            pcZF0(Hmatrix, Pt, K_b, K_e), subcarrier
        )

# -----------------------------
# Conventional ZF for comparison
# -----------------------------
covmap = compute_coverage_map_vectorized(h_freq, ZF_precoder0(H_b, Pt, K_b), subcarrier)
for sig_idx, sigma in enumerate(sigma2_lin):
    SE_eve[-1, sig_idx] = compute_sum_rate_eve(
        covmap, eve_ref, K_e - 1, sigma, num_subcarriers,
        ZF_precoder0(H_b, Pt, K_b), subcarrier
    )

# -----------------------------
# Save results to HDF5
# -----------------------------
filename = f"SE_eve_CC_f_1_NLOS_N_{train_channels.shape[0]}_D_{train_channels.shape[1]}.h5"
with h5py.File(filename, 'w') as f_se:
    f_se.create_dataset("SE", data=SE_eve)

#%% ====================================================================
# SR_epsilon Computation for Various NLDR and Benchmark Methods
# ====================================================================
# This section computes the secrecy rate (SR) with epsilon-security constraints
# for different NLDR-based methods, distance-insensitive neighbor selection,
# and conventional benchmarks. It iterates over SNR values and evaluates
# optimal power allocation for Eve's suppression.
# ====================================================================

# -----------------------------
# Hyperparameters for NLDR methods
# -----------------------------
max_neib_isomap = 5
max_dim_isomap = 9

max_neib_umap = 6
max_dim_umap = 10

max_neib_tsne = 14

# System parameters
num_subcarriers = 32
subcarrier = 16
Pt = 1  # Transmit power
fractions = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]  # Fraction of training set
N_tilde_values = [int(train_channels.shape[0] * f) for f in fractions]

# -----------------------------
# Bob and Eve positions
# -----------------------------
bob = [5311, 5440]  # Bob indices in training dataset
K_b = len(bob)      # Number of Bobs
eve_ref = 784       # Eve index in test dataset
eve_xy = test_positions[eve_ref]  # Eve position (x, y)

# -----------------------------
# Channel preparation
# -----------------------------
recovered_transposed = train_channels.reshape(train_channels.shape[0], num_subcarriers, L, M)
h_freq_recovered = recovered_transposed.permute(0, 2, 3, 1)
h_freq = h_freq_recovered.detach().cpu().numpy()

# Bob's channel matrix for selected subcarrier
H_b = np.reshape(h_freq[bob, :, :, subcarrier], (K_b, M * L))

# Noise variance for each SNR
snr = np.arange(-5, 36, 1)
sigma2_lin = np.mean(compute_sigma2_lin(H_b, snr, Pt), axis=0)

# -----------------------------
# Methods to evaluate
# -----------------------------
CM_methods = ["benchmark", "dist-insent", "isomap", "umap", "tsne"]
num_maps = len(CM_methods)
K_e = 33         # Number of eavesdroppers (including Eve)
eps = 1e-4       # Security constraint
omeg = 1e-6      # Small constant to avoid division by zero
SR = np.zeros((num_maps + 1, len(snr)))  # +1 for conventional ZF

# Reference neighbor channels for Eve
knn_ref, knn_pos_ref = get_eves_ref(eve_ref, K_e - 1, subcarrier)
H_e_ref = np.concatenate(
    (np.reshape(test_channels[eve_ref, subcarrier * M * L:(subcarrier + 1) * M * L], (1, M * L)), 
     knn_ref),
    axis=0
)

# -----------------------------
# Loop over each method
# -----------------------------
for i, method_name in enumerate(CM_methods):
    # --- Neighbor selection ---
    if method_name == "benchmark":
        knn_He, knn_pos = get_eves_ref(eve_ref, K_e - 1, subcarrier)
    elif method_name == "dist-insent":
        knn_He, knn_pos = get_eves_dist_insent(
            eve_ref, K_e - 1, subcarrier, N_tilde=N_tilde_values[-1]
        )
    else:
        # Charting-based NLDR methods
        if method_name == "isomap":
            knn_He, knn_pos = get_eves_CC(
                eve_ref, K_e - 1, subcarrier, method=method_name, 
                num_neib_ml=max_neib_isomap, dim_ls=max_dim_isomap, 
                N_tilde=N_tilde_values[-1]
            )
        elif method_name == "umap":
            knn_He, knn_pos = get_eves_CC(
                eve_ref, K_e - 1, subcarrier, method=method_name, 
                num_neib_ml=max_neib_umap, dim_ls=max_dim_umap, 
                N_tilde=N_tilde_values[-1]
            )
        elif method_name == "tsne":
            knn_He, knn_pos = get_eves_CC(
                eve_ref, K_e - 1, subcarrier, method=method_name, 
                num_neib_ml=max_neib_tsne, dim_ls=3, 
                N_tilde=N_tilde_values[-1]
            )

    # Concatenate Eve's reference channel with neighbors
    H_e = np.concatenate(
        (np.reshape(test_channels[eve_ref, subcarrier * M * L:(subcarrier + 1) * M * L], (1, M * L)), 
         knn_He),
        axis=0
    )

    # Complete channel matrix including Bob and Eve
    Hmatrix = np.concatenate((H_b, H_e), axis=0)

    # Compute interference power at Eve for security constraint
    nu_k = np.sum(np.square(np.abs(H_e_ref @ pcZF0(Hmatrix, Pt, K_b, K_e))), axis=1) / Pt

    # Compute secrecy rate for each SNR
    for sig_idx, sigma in enumerate(sigma2_lin):
        num = nu_k * Pt - (sigma * (np.power(2, (eps / K_e)) - 1))
        num[num < 0] = 0
        denum = nu_k + omeg * (np.power(2, (eps / K_e)) - 1)
        p_n = np.max(num / denum)          # Power allocated to Eve-suppressing signal
        p_u = Pt - p_n                     # Power allocated to user
        SR[i, sig_idx] = compute_sum_rate_bob(H_b, pcZF0(Hmatrix, p_u, K_b, K_e), sigma)

# -----------------------------
# Conventional ZF for comparison
# -----------------------------
nu_kzf = np.sum(np.square(np.abs(H_e_ref @ ZF_precoder0(H_b, Pt, K_b))), axis=1) / Pt
for sig_idx, sigma in enumerate(sigma2_lin):
    num = nu_kzf * Pt - (sigma * (np.power(2, (eps / K_e)) - 1))
    num[num < 0] = 0
    denum = nu_kzf + omeg * (np.power(2, (eps / K_e)) - 1)
    p_n = np.max(num / denum)
    p_u = Pt - p_n
    SR[-1, sig_idx] = compute_sum_rate_bob(H_b, ZF_precoder0(H_b, p_u, K_b), sigma)

# -----------------------------
# Save results to HDF5
# -----------------------------
filename = f"SR_CC_f_1_NLOS_N_{train_channels.shape[0]}_D_{train_channels.shape[1]}.h5"
with h5py.File(filename, 'w') as f_sr:
    f_sr.create_dataset("SR", data=SR)

#%% ====================================================================
# SR_epsilon with Dataset Subsampling
# ====================================================================
# This section evaluates the secrecy rate (SR) under epsilon-security constraints
# for different fractions of the training dataset. It compares benchmark, 
# distance-insensitive, and charting-based NLDR methods.
# ====================================================================

import numpy as np
import h5py

# -----------------------------
# Hyperparameters for NLDR methods
# -----------------------------
max_neib_isomap = 5
max_dim_isomap = 9

max_neib_umap = 6
max_dim_umap = 10

max_neib_tsne = 14

# System parameters
num_subcarriers = 32
subcarrier = 16
Pt = 1  # Transmit power

# Dataset fractions for subsampling
fractions = [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
N_tilde_values = [int(train_channels.shape[0] * f) for f in fractions]

# -----------------------------
# Bob and Eve positions
# -----------------------------
bob = [5311, 5440]  # Bob indices in training dataset
K_b = len(bob)      # Number of Bobs
eve_ref = 784       # Eve index in test dataset
eve_xy = test_positions[eve_ref]  # Eve position (x, y)

# -----------------------------
# Channel preparation
# -----------------------------
recovered_transposed = train_channels.reshape(train_channels.shape[0], num_subcarriers, L, M)
h_freq_recovered = recovered_transposed.permute(0, 2, 3, 1)
h_freq = h_freq_recovered.detach().cpu().numpy()

# Bob's channel matrix for selected subcarrier
H_b = np.reshape(h_freq[bob, :, :, subcarrier], (K_b, M * L))

# Noise variance for each SNR
snr = np.arange(-5, 36, 1)
sigma2_lin = np.mean(compute_sigma2_lin(H_b, snr, Pt), axis=0)

# -----------------------------
# Methods to evaluate
# -----------------------------
SR_tilde_methods = ["benchmark", "dist-insent", "umap"]  # Selected methods
num_maps = len(SR_tilde_methods)
K_e = 33      # Number of eavesdroppers (including Eve)
eps = 1e-4    # Security constraint
omeg = 1e-6   # Small constant to avoid division by zero

# Initialize SR storage: [dataset fraction, method, SNR]
SR_f = np.zeros((len(N_tilde_values), num_maps, len(snr)))

# Reference neighbor channels for Eve
knn_ref, knn_pos_ref = get_eves_ref(eve_ref, K_e - 1, subcarrier)
H_e_ref = np.concatenate(
    (np.reshape(test_channels[eve_ref, subcarrier * M * L:(subcarrier + 1) * M * L], (1, M * L)), 
     knn_ref),
    axis=0
)

# -----------------------------
# Loop over dataset fractions and methods
# -----------------------------
for n, n_tilde in enumerate(N_tilde_values):
    print(f"{fractions[n]*100:.0f}% of the dataset")
    for mn, methd_name in enumerate(SR_tilde_methods):
        print(f"Computing {methd_name} method")
        
        # --- Neighbor selection ---
        if methd_name == "benchmark":
            knn_He, knn_pos = get_eves_ref(eve_ref, K_e - 1, subcarrier)
        elif methd_name == "dist-insent":
            knn_He, knn_pos = get_eves_dist_insent(
                eve_ref, K_e - 1, subcarrier, N_tilde=n_tilde
            )
        else:
            # Charting-based NLDR methods
            if methd_name == "isomap":
                knn_He, knn_pos = get_eves_CC(
                    eve_ref, K_e - 1, subcarrier, method=methd_name,
                    num_neib_ml=max_neib_isomap, dim_ls=max_dim_isomap, 
                    N_tilde=n_tilde
                )
            elif methd_name == "umap":
                knn_He, knn_pos = get_eves_CC(
                    eve_ref, K_e - 1, subcarrier, method=methd_name,
                    num_neib_ml=max_neib_umap, dim_ls=max_dim_umap, 
                    N_tilde=n_tilde
                )
            elif methd_name == "tsne":
                knn_He, knn_pos = get_eves_CC(
                    eve_ref, K_e - 1, subcarrier, method=methd_name,
                    num_neib_ml=max_neib_tsne, dim_ls=3, N_tilde=n_tilde
                )

        # Concatenate Eve's reference channel with neighbors
        H_e = np.concatenate(
            (np.reshape(test_channels[eve_ref, subcarrier * M * L:(subcarrier + 1) * M * L], (1, M * L)), 
             knn_He),
            axis=0
        )

        # Complete channel matrix including Bob and Eve
        Hmatrix = np.concatenate((H_b, H_e), axis=0)

        # Compute interference power at Eve
        nu_k = np.sum(np.square(np.abs(H_e_ref @ pcZF0(Hmatrix, Pt, K_b, K_e))), axis=1) / Pt

        # Compute SR for each SNR
        for sig_idx, sigma in enumerate(sigma2_lin):
            num = nu_k * Pt - (sigma * (np.power(2, (eps / K_e)) - 1))
            num[num < 0] = 0
            denum = nu_k + omeg * (np.power(2, (eps / K_e)) - 1)
            p_n = np.max(num / denum)       # Power allocated to Eve-suppressing signal
            p_u = Pt - p_n                  # Power allocated to user
            SR_f[n, mn, sig_idx] = compute_sum_rate_bob(
                H_b, pcZF0(Hmatrix, p_u, K_b, K_e), sigma
            )

# -----------------------------
# Save results to HDF5
# -----------------------------
filename = f"SR_CC_Ke_{K_e}_NLOS_N_{train_channels.shape[0]}_D_{train_channels.shape[1]}.h5"
with h5py.File(filename, 'w') as f_fr:
    f_fr.create_dataset("SR", data=SR_f)
# %%
