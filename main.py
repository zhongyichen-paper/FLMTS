import numpy as np
from sklearn.model_selection import KFold
import RWR
import classifier as model
import os


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


# 将关联矩阵中部分的正样本改为负样本，调控数据集的正样本的Density,输入原始关联矩阵和输出关联矩阵中正样本的数量
def density_positive(data_ck, num_positive):
    if num_positive == data_ck.sum():
        return data_ck
    data_ck_ = np.zeros(shape=data_ck.shape)
    for i in range(data_ck.shape[0]):
        for j in range(data_ck.shape[1]):
            data_ck_[i, j] = data_ck[i, j]
    # 找出所有个 1
    pos_position = np.zeros(shape=(int(np.sum(data_ck)), 2))
    tmp_pos = 0
    for a in range(data_ck.shape[0]):
        for b in range(data_ck.shape[1]):
            if data_ck[a, b] == 1:
                pos_position[tmp_pos, 0] = a
                pos_position[tmp_pos, 1] = b
                tmp_pos += 1
    # print('关联矩阵中 1 的总个数： '+tmp_pos)  # tmp_pos =2141
    np.random.shuffle(pos_position)
    for i in range(tmp_pos - num_positive):  # 打乱后的正样本的位置，留下后面要保留正样本的个数
        row = int(pos_position[i, 0])
        col = int(pos_position[i, 1])
        data_ck_[row, col] = 0

    return data_ck_


# 获得所有样本在关联矩阵中的位置
def sample_position(data_ck):
    # 找出所有个 1
    pos_position = np.zeros(shape=(int(np.sum(data_ck)), 2))
    tmp_pos = 0
    for a in range(data_ck.shape[0]):
        for b in range(data_ck.shape[1]):
            if data_ck[a, b] == 1:
                pos_position[tmp_pos, 0] = a
                pos_position[tmp_pos, 1] = b
                tmp_pos += 1
    print('关联矩阵中 1 的总个数： ', tmp_pos)  # tmp_pos =2141

    # 找出所有的 0
    N = int(data_ck.shape[0] * data_ck.shape[1])
    pos_position_2 = np.zeros(shape=(int(N - tmp_pos), 2))  # 这里的tmp_pos 可以后面用sum.(data_ck)替代，前面就可以不用遍历找个数了
    tmp_pos_2 = 0
    for a in range(data_ck.shape[0]):
        for b in range(data_ck.shape[1]):
            if data_ck[a, b] == 0:
                pos_position_2[tmp_pos_2, 0] = a
                pos_position_2[tmp_pos_2, 1] = b
                tmp_pos_2 += 1

    position = np.append(pos_position, pos_position_2, axis=0)
    print('Global sample: ', end=' ')

    return position


# 将训练和测试数据合成,传入的分别是训练集位置，测试集位置，化合物特征矩阵，激酶特征矩阵,关联矩阵
def data_sample(train_pos, test_pos, data_cf, data_kf, data_ck):
    col_arr = data_cf.shape[1] + data_kf.shape[1] + 1  # 这里是合并矩阵列向量的维数，将向量合并还要加上一位 lable
    # 训练集的数据合成
    train_data = np.zeros(shape=(len(train_pos), col_arr))
    tr_count = 0  # 用于记录训练集中的正样本的个数
    for i in range(len(train_pos)):
        c = int(train_pos[i, 0])
        k = int(train_pos[i, 1])
        train_data[i, col_arr - 1] = data_ck[c, k]  # 将关联矩阵的标签赋给数据的最后一位
        if data_ck[c, k] == 1:
            tr_count += 1
        for j in range(col_arr - 1):  # 将化合物特征和激酶的特征合并
            if j < data_cf.shape[1]:
                train_data[i, j] = data_cf[c, j]
            else:
                train_data[i, j] = data_kf[k, j - data_cf.shape[1]]

    # 测试集的数据合成
    test_data = np.zeros(shape=(len(test_pos), col_arr))
    te_count = 0  # 用于记录测试集中正样本的个数
    for i in range(len(test_pos)):
        c = int(test_pos[i, 0])
        k = int(test_pos[i, 1])
        test_data[i, col_arr - 1] = data_ck[c, k]  # 将关联矩阵的标签赋给数据的最后一位
        if data_ck[c, k] == 1:
            te_count += 1
        for j in range(col_arr - 1):  # 将化合物特征和激酶的特征合并
            if j < data_cf.shape[1]:
                test_data[i, j] = data_cf[c, j]
            else:
                test_data[i, j] = data_kf[k, j - data_cf.shape[1]]
    print('num of positive sample in train: {0}, num of positive sample in test: {1}'.format(tr_count, te_count))

    return [train_data, test_data]


# 用于交叉验证，传入的分别是所有样本数据的位置，化合物的特征，激酶的特征，关联矩阵
def cross_validation(position, data_ck_den, restartPro):
    Y = np.zeros(len(position))
    for i in range(len(Y)):
        Y[i] = i
    np.random.shuffle(position)  # 将二维数组按行打乱顺序
    kf = KFold(n_splits=5)
    AUC = np.zeros(5)
    AUPR = np.zeros(5)
    F1 = np.zeros(5)
    Balanced_accuracy = np.zeros(5)
    Recall = np.zeros(5)
    Precision = np.zeros(5)
    i = 0
    # 5折交叉验证，将数据分为5份，每次取一份作为test集
    for train_index, test_index in kf.split(Y):  # 这里 Y 被平均分为了5 分，依次选取一份作为测试集
        train_pos = position[train_index]
        test_pos = position[test_index]

        new_fea_com, new_fea_kin = RWR.main(restartPro, test_pos, data_ck_den)  # 这里用重启返回新的特征矩阵
        train_data, test_data = data_sample(train_pos, test_pos, new_fea_com, new_fea_kin, data_ck_den)

        x_train, label_train = np.split(train_data, (train_data.shape[1] - 1,), axis=1)  # 将数组train_Y 按列分割，分割线为第332列
        x_test, label_test = np.split(test_data, (test_data.shape[1] - 1,), axis=1)

        fpr, tpr, roc_auc, re, pr, aupr, recall, precision, f1_score, BA = model.train_RF(x_train, label_train, x_test,
                                                                                          label_test)

        AUC[i] = roc_auc
        AUPR[i] = aupr
        Recall[i] = recall
        Precision[i] = precision
        F1[i] = f1_score
        Balanced_accuracy[i] = BA
        i += 1

        if i == 5:
            print("PKIS RF AUC: {:.4f}, AUPR: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, BA: {:.4f}"
                  .format(AUC.mean(), AUPR.mean(), Recall.mean(), Precision.mean(), F1.mean(), Balanced_accuracy.mean()))

    return AUC.mean(), AUPR.mean(), Recall.mean(), Precision.mean(), F1.mean(), Balanced_accuracy.mean()


def main(restartPro):
    path = os.path.dirname(os.getcwd())
    data_ck = read_data(path + '/data/PKIS_Matrix_Compound_Kinase2.txt')
    data_ck = np.array(data_ck)
    den = [2414]
    for de in den:
        AUC = np.zeros(10)
        AUPR = np.zeros(10)
        F1 = np.zeros(10)
        Balanced_accuracy = np.zeros(10)
        Recall = np.zeros(10)
        Precision = np.zeros(10)
        for i in range(10):
            data_ck_den = density_positive(data_ck, de)
            print('num of positive sample in data sets: ', end=' ')

            position = sample_position(data_ck_den)
            auc, aupr, recall, pre, f1, BA = cross_validation(position, data_ck_den, restartPro)
            AUC[i] = auc
            AUPR[i] = aupr
            Recall[i] = recall
            Precision[i] = pre
            F1[i] = f1
            Balanced_accuracy[i] = BA
        print('**********************' * 3)
        print("{} RF AUC: {:.4f}, AUPR: {:.4f}, Recall: {:.4f}, Precision: {:.4f}, F1: {:.4f}, BA: {:.4f}"
              .format('PKIS', AUC.mean(), AUPR.mean(), Recall.mean(), Precision.mean(), F1.mean(),
                      Balanced_accuracy.mean()))
        print()


if __name__ == '__main__':
    # R = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    R = [0.7]
    for i in R:
        main(i)
