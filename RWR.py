import os.path

import numpy as np


# Read data
def read_data(path):
    data = []
    for line in open(path, 'r'):
        ele = line.strip().split(" ")
        tmp = []
        for e in ele:
            if e != '':
                tmp.append(float(e))
        data.append(tmp)
    return data


# 将矩阵按行归一化
def normalize(w):
    for row in range(w.shape[0]):  # 标准化上三角的值
        max = w[row, row]
        for col in range(row, w.shape[1]):
            w[row, col] = w[row, col] / max

    for row in range(w.shape[0]):  # 将上三角的值赋给下三角，形成对称矩阵
        for col in range(row):
            w[row, col] = w[col, row]

    if w.max() > 1:
        print('error: 矩阵标准化出错！！！！', w.max())

    return w


# 重启随机游走算法
def diffusionRWR(A, maxiter, restartProb, netname):
    n = len(A)
    # normalize the adjacency matrix
    P = A / A.sum(axis=0)  # 按列求和

    restart = np.eye(n)
    Q = np.eye(n)
    for i in range(1, maxiter):
        Q_new = (1 - restartProb) * np.dot(P, Q) + restart * restartProb
        delta = np.linalg.norm((Q - Q_new))  # 计算F范数，表示误差
        Q = Q_new
        if delta < 1e-16:
            break
    return Q


# 在异构网络上的重启随机游走，通过矩阵拼接组成异构网，输入激酶相似性矩阵，化合物相似性矩阵，关联矩阵，最大迭代次数，重启概率
def diffusionRWR_heter(matrix_c, matrix_k, matrix_link, maxIter, restartProb):
    matrix_link_T = matrix_link.T
    matrix1 = np.append(matrix_c, matrix_link, axis=1)
    matrix2 = np.append(matrix_link_T, matrix_k, axis=1)
    Matrix = np.append(matrix1, matrix2, axis=0)
    M = diffusionRWR(Matrix, maxIter, restartProb, 'heternetwork')

    for row in range(matrix_c.shape[0]):
        for col in range(matrix_c.shape[1]):
            matrix_c[row, col] = M[row, col]

    for row in range(matrix_c.shape[0], M.shape[0]):
        for col in range(matrix_c.shape[1], M.shape[1]):
            matrix_k[row-matrix_c.shape[0], col-matrix_c.shape[1]] = M[row, col]

    return [matrix_c, matrix_k]


def main(restartPro, test_position, data_ck_den):
    path = os.path.dirname(os.getcwd())
    data_com_sim = read_data(path + '/data/PKIS_ComComSim_167.txt')
    data_com_sim = np.array(data_com_sim)

    data_pro_sim = read_data(path + '/data/PKIS_KinKinSim.txt')
    data_pro_sim = np.array(data_pro_sim)
    data_pro_sim = normalize(data_pro_sim)  #

    data_link = np.zeros_like(data_ck_den)  # 直接在data_ck_den上改，会影响
    for i in range(data_ck_den.shape[0]):
        for j in range(data_ck_den.shape[1]):
            data_link[i, j] = data_ck_den[i, j]

    for i in range(len(test_position)):
        row = int(test_position[i, 0])
        col = int(test_position[i, 1])
        data_link[row, col] = 0  # 将测试集的边全部变为零

    score_com, score_pro = diffusionRWR_heter(data_com_sim, data_pro_sim, data_link, 100, restartPro)
    return [score_com, score_pro]


if __name__ == '__main__':
    main(0.1, '', '')
