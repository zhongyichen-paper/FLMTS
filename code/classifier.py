# RF
import argparse
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import metrics


# 画图
def show(fpr, tpr, auc, re, pr, aupr):
    plt.figure(figsize=(6, 12))

    plt.subplot(121)
    plt.title('Receiver Operating Characteristic(ROC)')
    plt.plot(fpr, tpr, 'b', label='PKIS_AUC = %0.3f' % auc)
    plt.plot([-0.1, 1], [-0.1, 1], 'r--')
    plt.xlim([-0.1, 1])
    plt.ylim([-0.1, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc='lower right')

    plt.subplot(122)
    plt.title('Precision-Recall curve ')
    plt.plot(re, pr, 'b', label='PKIS_AUPR = %0.3f' % aupr)
    # plt.plot([0, 1.1], [0, 1.1], 'r--')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel('recall(%)')
    plt.ylabel('precision(%)')
    plt.legend(loc='upper right')  # 图例位置

    plt.show()


# RF
def train_RF(x_train, label_train, x_test, label_test):
    # n_estimators 森林中树的数量
    model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=2, random_state=0, n_jobs=3)
    model.fit(x_train, label_train.ravel())
    y_prob = model.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = metrics.roc_curve(label_test.ravel(), y_prob)
    roc_auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(label_test.ravel(), y_prob)  # precision recall 是数组
    aupr = metrics.auc(recall, precision)

    y_pre = np.zeros(shape=(len(y_prob), 1))
    for m in range(int(len(y_prob) * 0.10)):
        max = 0
        p = 0
        for j in range(len(y_prob)):
            if y_prob[j] > max:
                max = y_prob[j]
                p = j
        y_prob[p] = -1
        y_pre[p, 0] = 1

    f1_score = metrics.f1_score(label_test.ravel(), y_pre)
    recall1 = metrics.recall_score(label_test.ravel(), y_pre)
    BA = metrics.balanced_accuracy_score(label_test.ravel(), y_pre)  # 非平衡准确率
    precision1 = metrics.precision_score(label_test.ravel(), y_pre)
    print("RF  AUC:  probability{0}, prediction{1} AUPR{1}: ".format(roc_auc, aupr))
    print("RF  F1-score{0}".format(f1_score))
    print('RF recall1:{0}, accuracy{1}, precision1{2} '.format(recall1, BA, precision1))

    return [fpr, tpr, roc_auc, recall, precision, aupr,
            recall1, precision1, f1_score, BA]  # 每次返回的fpr,tpr长度不一样


if __name__ == '__main__':
    # RF 训练
    train_RF(100)
