import numpy as np
import logging
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import Matern
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.spatial import distance_matrix

logger = logging.getLogger(__name__)


class SMOPCA:
    def __init__(self, Z_dim, Y_list):
        """
        :param Z_dim: dimension of the latent factors
        :param Y_list: list of input modalities, should be all in the shape of (#feats, #cells)
        """
        assert all(Y.shape[1] == Y_list[0].shape[1] for Y in Y_list)
        self.Y_list = Y_list
        self.m_list = [Y.shape[0] for Y in Y_list]
        self.n = Y_list[0].shape[1]
        self.d = Z_dim
        self.modality_num = len(Y_list)

        # simplified for easier inference
        self.q_list = [1 for _ in range(len(Y_list))]
        self.X_list = [np.ones((self.n, 1)) for _ in range(len(Y_list))]
        self.M_list = [np.identity(self.n) - X @ np.linalg.inv((X.T @ X)) @ X.T for X in self.X_list]

        self.K = None
        self.K_inv = None
        self.Z = None
        self.W_hat_list = []
        self.sigma_hat_sqr_list = []
        logger.info(f"SMOPCA object created, with {self.n} cells and {[Y.shape[0] for Y in self.Y_list]} features")

    def buildKernel(self, pos=None, kernel_type="matern", method="sklearn", nu=1.5, length_scale=1.0):
        """
        :param pos: location coordinates for calculating cell-cell similarity
        :param kernel_type: type of kernel, recommend matern when cell number is large
        :param method: implementation of gaussian kernel, recommend sklearn
        :param nu: matern kernel parameter
        :param length_scale: matern kernel length scale, or gaussian / tsne kernel gamma, or cauchy kernel sigma
        """
        if kernel_type == "gaussian":
            logger.info(f"calculating {kernel_type} kernel with {method} implementation, gamma = {length_scale}")
            if method == "sklearn":
                self.K = rbf_kernel(pos, pos, gamma=1 / length_scale)
            elif method == "scipy":
                self.K = np.exp(-np.power(distance_matrix(pos, pos), 2) / length_scale)
        elif kernel_type == 'matern':
            logger.info(f"calculating {kernel_type} kernel, nu = {nu}, length_scale = {length_scale}")
            matern_obj = Matern(length_scale=length_scale, nu=nu)
            self.K = matern_obj(X=pos, Y=pos)
        elif kernel_type == 'cauchy':
            logger.info(f"calculating {kernel_type} kernel, sigma = {length_scale}")
            squared_diff = distance_matrix(pos, pos) ** 2
            self.K = 1 / (1 + squared_diff / length_scale ** 2)
        elif kernel_type == "tsne":
            pos *= length_scale
            self.K = np.power(np.power(distance_matrix(pos, pos), 2) + 1, -1)
        elif kernel_type == "dummy":
            logger.info("using Identity as the kernel matrix")
            self.K = np.identity(self.n)
        else:
            logger.error("other kernel type not implemented yet!")
            raise NotImplemented
        logger.debug("calculating kernel inverse")
        self.K_inv = np.linalg.inv(self.K)

        # check numerical stability
        K_inv_det, K_num, KK_inv_det = np.linalg.det(self.K), np.sum(self.K - np.identity(self.n)), np.linalg.det(self.K @ self.K_inv)
        if KK_inv_det < -1 or KK_inv_det > 1000:
            logger.warning("kernel matrix status: det={:.4f}, K_num={:.4f}, det(KKinv)={:.4f}\n"
                           "numerical instability is expected, please try smaller gamma or length_scale".format(K_inv_det, K_num, KK_inv_det))
        else:
            logger.debug("kernel matrix status: det={:.4f}, K_num={:.4f}, det(KKinv)={:.4f}".format(
                np.linalg.det(self.K), np.sum(self.K - np.identity(self.n)), np.linalg.det(self.K @ self.K_inv)
            ))

    def estimateSigmaW(self, iterations=20, tol=1e-5, sigma_init_list=(), sigma_xtol_list=()):
        """
        :param iterations: number of outer iterations for estimation
        :param tol: tolerance for sigma estimation
        :param sigma_init_list: init value for sigma, should include the same number of values as the number of modalities
        :param sigma_xtol_list: xtol parameter for brentq function, should include the same number of values as the number of modalities
        """
        assert len(sigma_init_list) == self.modality_num
        assert len(sigma_xtol_list) == self.modality_num
        logger.info("start estimating parameters, this will take a while...")
        bound_list = [None for _ in range(self.modality_num)]
        for modality in range(self.modality_num):
            Y = self.Y_list[modality]
            M = self.M_list[modality]
            tr_YMY_T = np.trace(Y @ M @ Y.T)
            YM = Y @ M
            MY_T = M @ Y.T
            MK = M @ self.K
            In = np.identity(self.n)
            sigma_sqr = sigma_init_list[modality]
            sigma_hat_sqr = None
            W_hat = None
            logger.info(f"estimating sigma{modality + 1}")
            for iteration in range(iterations):
                G = YM @ np.linalg.inv(M + sigma_sqr * self.K_inv) @ MY_T
                vals, vec = eigh(G)
                W_hat = vec[:, -self.d:]  # first d eigenvectors
                assert W_hat.shape == (self.m_list[modality], self.d)

                def jac_sigma_sqr(_sigma_sqr):  # derivative of -log likelihood
                    part1 = self.m_list[modality] * (self.n - self.q_list[modality]) / _sigma_sqr
                    MKOverXPlusInInv = np.linalg.inv(MK / _sigma_sqr + In)
                    part2 = -self.d * np.trace(MK @ MKOverXPlusInInv) / (_sigma_sqr ** 2)
                    MPlusXKInvInv = np.linalg.inv(M + _sigma_sqr * self.K_inv)
                    P = _sigma_sqr * MPlusXKInvInv @ self.K_inv @ MPlusXKInvInv + MPlusXKInvInv
                    sumQuadraticP3 = 0
                    for l in range(self.d):
                        wl = W_hat[:, l]
                        MY_T_wl = MY_T @ wl
                        sumQuadraticP3 += MY_T_wl.T @ P @ MY_T_wl
                    part3 = (sumQuadraticP3 - tr_YMY_T) / (_sigma_sqr ** 2)
                    res = part1 + part2 + part3
                    logger.debug("jac{}({:.5f}) = {:.5f}".format(modality + 1, _sigma_sqr, res))
                    return res

                # estimate a bound for tighter searching range
                if bound_list[modality] is None:
                    lb = ub = 0.1
                    lb_res = -np.inf
                    ub_res = np.inf
                    for sigma in np.arange(0.1, 10.0, 0.1):
                        res = jac_sigma_sqr(sigma)
                        if res < 0:
                            lb = sigma
                            lb_res = res
                        else:
                            ub = sigma
                            ub_res = res
                            break
                    if abs(lb_res) < 1000:  # for a safer bound since this is a bound dependent on last iteration (init values)
                        lb -= 0.05
                    if abs(ub_res) < 1000:
                        ub += 0.05
                    bound_list[modality] = (lb, ub)
                    logger.info("sigma{} using bound: ({:.5f}, {:.5f})".format(modality + 1, lb, ub))

                sigma_hat_sqr = brentq(jac_sigma_sqr, bound_list[modality][0], bound_list[modality][1], xtol=sigma_xtol_list[modality])
                logger.info("iter {} sigma{} brentq done, sigma{}sqr = {:.5f}, sigma{}hatsqr = {:.5f}".format(
                    iteration, modality + 1, modality + 1, sigma_sqr, modality + 1, sigma_hat_sqr))
                if abs(sigma_sqr - sigma_hat_sqr) < tol:
                    logger.info(f"reach tolerance threshold, sigma{modality + 1} done!")
                    self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                    self.W_hat_list.append(W_hat)
                    break
                sigma_sqr = sigma_hat_sqr
                if iteration == iterations - 1:
                    logger.warning(f"iteration not enough for sigma{modality + 1}!")
                    self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                    self.W_hat_list.append(W_hat)
        logger.info("estimation complete!")
        for modality, sigma_hat_sqr in enumerate(self.sigma_hat_sqr_list):
            logger.info("sigma{}hatsqr = {:.5f}".format(modality + 1, sigma_hat_sqr))

    def calculatePosterior(self):
        logger.info("calculating posterior")
        Z = np.zeros((self.d, self.n))
        A = self.K_inv
        for modality in range(self.modality_num):
            A += (self.M_list[modality] / self.sigma_hat_sqr_list[modality])
        A *= 0.5
        A_inv = np.linalg.inv(A)
        for l in range(self.d):
            bl = 0
            for modality in range(self.modality_num):
                w_l = self.W_hat_list[modality][:, l]
                p = self.M_list[modality] @ self.Y_list[modality].T @ w_l / self.sigma_hat_sqr_list[modality]
                bl += p
            Zl = 0.5 * A_inv @ bl
            Z[l, :] = Zl
        return Z
