"""
Based on "Disentangling by Factorising" (https://github.com/nmichlo/disent/blob/main/disent/metrics/_sap.py).
"""
import logging
import torch
from sklearn import svm
import numpy as np
import pdb
from src.disentangle_metrics.utils import latents_and_factors

logger = logging.getLogger(__name__)

def compute_score_matrix(train_latents, train_factors, test_latents, test_factors, continuous_factors):
    num_latents = train_latents.shape[0]
    num_factors = train_factors.shape[0]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            train_latents_i = train_latents[i, :]
            train_factors_j = train_factors[j, :]
            if continuous_factors:
                # Attribute is considered continuous.
                cov_mu_i_y_j = np.cov(train_latents_i, train_factors_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                else:
                    score_matrix[i, j] = 0.
            else:
                # Attribute is considered discrete.
                test_latents_i = test_latents[i, :]
                test_factors_j = test_factors[j, :]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(train_latents_i[:, np.newaxis], train_factors_j)
                pred = classifier.predict(test_latents_i[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == test_factors_j)
    return score_matrix

def compute_avg_diff_top_two(score_matrix):
    sorted_matrix = np.sort(score_matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])

def compute_sap(train_latents, train_factors, test_latents, test_factors, continuous_factors):
    score_matrix = compute_score_matrix(train_latents, train_factors, test_latents, test_factors, continuous_factors)
    assert score_matrix.shape[0] == train_latents.shape[0]
    assert score_matrix.shape[1] == train_factors.shape[0]

    sap_score = compute_avg_diff_top_two(score_matrix)

    return sap_score

def SAP(dataset, model, batch_size, num_train, num_test, loss_fn, continuous_factors=False):
    logger.debug("*********************Generating training set*********************")
    model.eval()
    with torch.no_grad():
        train_latents, train_factors = latents_and_factors(dataset, model, batch_size, num_train, loss_fn)
        test_latents, test_factors = latents_and_factors(dataset, model, batch_size, num_test, loss_fn)
    return compute_sap(train_latents, train_factors, test_latents, test_factors, continuous_factors)


