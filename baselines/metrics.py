import faiss
import numpy as np
from tqdm import tqdm


def f1_score(dataset, dist_mat, idx_mat, threshold):
    f1_score = []
    for i in range(len(dataset)):  # for each product
        target = set(np.where(dataset['category'] == dataset['category'].iloc[i])[0])  # set of products in the same category
        pred = set(idx_mat[i, np.where(dist_mat[i] < threshold)[0]])  # set of products such that the distance is less than the threshold
        tp = len(target.intersection(pred))  # number of products correctly predicted to be similar to the current product
        score = 2 * tp / (len(target) + len(pred))  # f1 score
        f1_score.append(score)
    return np.mean(f1_score)


def best_f1_score(dataset, dist_mat, idx_mat):
    mx = dist_mat[:, 50].min()
    mn = 0
    mx_score = 0
    mx_threshold = 0

    for i in tqdm(range(21), desc="Finding best threshold"):
        threshold = i * ((mx - mn) / 20) + mn
        f1 = f1_score(dataset, dist_mat, idx_mat, threshold)
        if f1 > mx_score:
            mx_score = f1
            mx_threshold = threshold
    
    return mx_score, mx_threshold


def _dcg(pred, target):
    return np.sum(np.isin(pred, target) / np.log2(np.arange(2, 2 + len(pred))))


def _idcg(target):
    return np.sum(1 / np.log2(np.arange(2, 2 + len(target))))


def ndcg(dataset, idx_mat):
    ndcg = []
    for i in range(len(dataset)):  # for each product
        target = np.where(dataset['category'] == dataset['category'].iloc[i])[0]  # set of products that have the same label group
        pred = idx_mat[i]  # set of products predicted to be in the same group
        ndcg.append(_dcg(pred, target) / _idcg(target))
    return np.mean(ndcg)

def neighbours(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    for e in embeddings:
        index.add(e.reshape(1, -1))
    dist_mat, idx_mat = index.search(embeddings, 51)
    return dist_mat, idx_mat
