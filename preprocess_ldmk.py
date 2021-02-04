import os
import pickle
import numpy as np
from sklearn.decomposition import PCA


def get_pca(landmarks):
    pca = PCA(n_components=20)  # landmarks PCA
    pca.fit(landmarks)
    return pca


if __name__ == '__main__':
    root = './lists/mouth_ldmk.txt'
    out_pca = './lists/pca.pickle'
    f = open(root, 'r')
    fpca = open(out_pca, 'wb')
    lines = f.readlines()
    landmarks = []
    for line in lines:
        ldmks = line.strip().split()[:-1]
        real_ldmks = [float(i) for i in ldmks]
        landmarks.append(real_ldmks)
    landmarks = np.stack(landmarks, axis=0)
    landmarks.astype('float64')
    pca = get_pca(landmarks)
    n_landmarks = pca.transform(landmarks)
    pickle.dump(pca,fpca)
