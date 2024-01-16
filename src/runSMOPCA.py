import logging
import h5py
import numpy as np
import warnings
from utils import clustering_metric, preprocess_hvg, plot_cluster, color_spatialciteseq
from SMOPCA import SMOPCA
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)


def run_SMOPCA_pbmc_sim(rep=0, length_scale=5, kernel_type='matern'):
    data_file = h5py.File(f"../../data/PBMC/sim/PBMC_sim_rep{rep}.h5", 'r')
    X1 = np.array(data_file['X1'])
    X2 = np.array(data_file['X2'])
    pos = np.array(data_file['pos'])
    y = np.array(data_file['Y'])
    data_file.close()
    X1, X2 = preprocess_hvg(x_list=[X1, X2], select_list=[True, False], top=1000)
    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20)
    smopca.buildKernel(pos=pos, kernel_type=kernel_type, length_scale=length_scale)
    smopca.estimateSigmaW(sigma_init_list=(1, 1), tol=2e-5, sigma_xtol_list=(1e-6, 1e-6))
    z = smopca.calculatePosterior()
    y_pred = KMeans(n_clusters=len(np.unique(y)), n_init=100).fit_predict(z.T)
    ami, nmi, ari = clustering_metric(y, y_pred)
    print("AMI={}, NMI={}, ARI={}".format(ami, nmi, ari))


def run_SMOPCA_citeseq():
    # data_file = h5py.File("../data/RealData/CITEseq/PBMC.h5", 'r')
    data_file = h5py.File("../data/RealData/CITEseq/SLN208D2.h5", 'r')
    X1 = np.array(data_file['X1'])
    X2 = np.array(data_file['X2'])
    pos = np.array(data_file['pos'])
    y = np.array(data_file['Y'])
    data_file.close()
    X1, X2 = preprocess_hvg(x_list=[X1, X2], select_list=[True, False], top=1000)
    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20)
    smopca.buildKernel(pos=pos, kernel_type='matern', length_scale=0.25)
    smopca.estimateSigmaW(sigma_init_list=(1, 1), tol=2e-5, sigma_xtol_list=(1e-6, 1e-6))
    z = smopca.calculatePosterior()
    y_pred = KMeans(n_clusters=len(np.unique(y)), n_init=100).fit_predict(z.T)
    ami, nmi, ari = clustering_metric(y, y_pred)
    print("AMI={}, NMI={}, ARI={}".format(ami, nmi, ari))


def run_SMOPCA_smageseq():
    # data_file = h5py.File("../data/RealData/SMAGEseq/PBMC_3k.h5", 'r')
    data_file = h5py.File("../data/RealData/SMAGEseq/PBMC_10k.h5", 'r')
    X1 = np.array(data_file['X1'])
    X2 = np.array(data_file['X2'])
    pos = np.array(data_file['pos'])
    y = np.array(data_file['Y'])
    data_file.close()
    X1, X2 = preprocess_hvg(x_list=[X1, X2], select_list=[True, True], top=1000)
    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20)
    smopca.buildKernel(pos=pos, kernel_type='matern', length_scale=0.25)
    smopca.estimateSigmaW(sigma_init_list=(1, 1), tol=2e-5, sigma_xtol_list=(1e-6, 1e-6))
    z = smopca.calculatePosterior()
    y_pred = KMeans(n_clusters=len(np.unique(y)), n_init=100).fit_predict(z.T)
    ami, nmi, ari = clustering_metric(y, y_pred)
    print("AMI={}, NMI={}, ARI={}".format(ami, nmi, ari))


def run_SMOPCA_spatialciteseq():
    color_spatialciteseq = ['#89c3e7', '#fee215', '#e6194b', '#3cb44b', '#911eb4', '#0f84c4', '#f58231']
    data_file = h5py.File("../data/RealData/SpatialCITEseq/Humantonsil_filtered.h5", 'r')
    X1 = np.array(data_file['normalized_svg_count'])
    X2 = np.array(data_file['normalized_protein_count'])
    pos = np.array(data_file['pos'])
    pos[:, 1] *= -1
    data_file.close()
    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20)
    smopca.buildKernel(pos=pos, kernel_type='matern', length_scale=5)
    smopca.estimateSigmaW(sigma_init_list=(1, 1), tol=2e-5, sigma_xtol_list=(1e-6, 1e-6))
    z = smopca.calculatePosterior()
    y_pred = KMeans(n_clusters=7, n_init=100).fit_predict(z.T)
    plot_cluster(labels=y_pred, pos=pos, colorList=color_spatialciteseq, pointSize=2)


def run_SMOPCA_misarseq():
    data_file = h5py.File("../data/RealData/MISARseq/MISAR_seq_mouse_E15_brain.h5", 'r')
    X1 = np.array(data_file['normalized_svg_gene_count'])
    X2 = np.array(data_file['normalized_svg_mapped_gene_count'])
    pos = np.array(data_file['pos'])
    pos[:, 1] *= -1
    y = np.array(data_file['Y'])
    data_file.close()
    smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20)
    smopca.buildKernel(pos=pos, kernel_type='matern', length_scale=5)
    smopca.estimateSigmaW(sigma_init_list=(1, 1), tol=2e-5, sigma_xtol_list=(1e-6, 1e-6))
    z = smopca.calculatePosterior()
    y_pred = KMeans(n_clusters=len(np.unique(y)), n_init=100).fit_predict(z.T)
    ami, nmi, ari = clustering_metric(y, y_pred)
    print("AMI={}, NMI={}, ARI={}".format(ami, nmi, ari))


run_SMOPCA_spatialciteseq()
