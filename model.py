import data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import csv
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import random
random.seed(0)


X, y = data.read_csv()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# <editor-fold desc="CLASSIFIERS">
# 单分类器，9种
KNN = KNeighborsClassifier()
RF = RandomForestClassifier()  # Bagging
LR = LogisticRegression(max_iter=10000)
DT = DecisionTreeClassifier()
ET = ExtraTreesClassifier()
GNB = GaussianNB()
SVM = SVC(probability=True)
LDA = LinearDiscriminantAnalysis()  # 线性判别分析
MLP = MLPClassifier(max_iter=10000)  # 神经网络
clfList = [KNN, RF, LR, DT, ET, GNB, SVM, LDA, MLP]
clfListStr = ['KNN', 'RF', 'LR', 'DT', 'ET', 'GNB', 'SVM', 'LDA', 'MLP']

# Bagging同构集成，8种，是人是鬼都能bagging
bagKNN = BaggingClassifier(base_estimator=KNN, n_estimators=50)
bagRF = BaggingClassifier(base_estimator=RF, n_estimators=10)
bagLR = BaggingClassifier(base_estimator=LR, n_estimators=50)
bagDT = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50)
bagET = BaggingClassifier(base_estimator=ET, n_estimators=50)
bagGNB = BaggingClassifier(base_estimator=GNB, n_estimators=50)
bagSVM = BaggingClassifier(base_estimator=SVM, n_estimators=50)
bagLDA = BaggingClassifier(base_estimator=LDA, n_estimators=50)
bagClfList = [RF, ET, bagRF, bagLR, bagET, bagSVM]
bagClfListStr = ['RF', 'ET', 'bagRF', 'bagLR', 'bagET', 'bagSVM']

# AdaBoost同构集成，4种
boostRF = AdaBoostClassifier(base_estimator=RF, n_estimators=20)
boostLR = AdaBoostClassifier(base_estimator=LR, n_estimators=20)
boostET = AdaBoostClassifier(base_estimator=ET, n_estimators=20)
boostSVM = AdaBoostClassifier(base_estimator=SVM, n_estimators=20)
boostClfList = [boostRF, boostLR, boostET, boostSVM]
boostClfListStr = ['boostRF', 'boostLR', 'boostET', 'boostSVM']

# Stacking异质集成，2种
stackList1 = [RF, LR, ET, SVM]
stack1 = StackingClassifier(classifiers=stackList1, use_probas=True,  average_probas=False, meta_classifier=LR)

stackList2 = [boostLR, boostSVM]
stack2 = StackingClassifier(classifiers=stackList2, use_probas=True,  average_probas=False, meta_classifier=LR)

stackClfList = [stack1, stack2]
stackClfListStr = ['stackList1', 'stackList2']
# </editor-fold>


# 获取auc得分，青春版验证方法
def get_score_auc(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')  # 交叉验证
    return scores.mean()


# 调参时过程函数，弱分类器调参意义不大，1.集成学习不需要弱分类器有多强；2.调参结果不能对所有药品都适用；3.效果与随机数种子有关，不能保证哪个就最好
def optimization():
    scores = []
    k_range = range(1, 50)
    for i in k_range:
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(X_train, y_train)
        score = get_score_auc(clf)
        scores.append(score)
        print(i)

    plt.plot(k_range, scores)
    plt.show()


# 评价指标
def get_score(clf_list, clf_list_str):
    rst_data = pd.DataFrame(columns=['DrugID', 'auc', 'auc_std', 'accuracy', 'accuracy_std', 'precision',
                                     'precision_std', 'recall', 'recall_std', 'f1', 'f1_std', 'Method'])
    for clf, label in zip(clf_list, clf_list_str):
        scoring = {'roc_auc', 'accuracy', 'precision', 'f1', 'recall'}
        scores = model_selection.cross_validate(clf, X, y, cv=10, scoring=scoring)
        # print("accuracy: %0.3f (+/- %0.3f) [%s]" % (scores.mean(), scores.std(), label))
        print(label, scores['test_roc_auc'].mean())
        # 循环训练时将auc输出
        new_row = {'DrugID': data.drugID,
                   'auc': scores['test_roc_auc'].mean(), 'auc_std': scores['test_roc_auc'].std(),
                   'accuracy': scores['test_accuracy'].mean(), 'accuracy_std': scores['test_accuracy'].std(),
                   'precision': scores['test_precision'].mean(), 'precision_std': scores['test_precision'].std(),
                   'recall': scores['test_recall'].mean(), 'recall_std': scores['test_recall'].std(),
                   'f1': scores['test_f1'].mean(), 'f1_std': scores['test_f1'].std(),
                   'Method': label}
        rst_data = data.append(new_row, ignore_index=True)
    return rst_data


# 写入文件
def write_to_file(rst_data):
    print(rst_data)
    out_path = data.source + "/Result/" + data.drugID + ".csv"
    rst_data.to_csv(out_path, sep=',', index=True, header=True)


# 画ROC曲线
def draw_roc(clf, clfStr, X, y):
    tpr_list = []
    auc_list = []
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    i = 0
    for i, (train, test) in enumerate(cv.split(X, y)):
        model = clf
        model.fit(X.iloc[train], y.iloc[train])
        probas_ = model.predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])   # 真阳率/精确率、伪阳率/召回率、临界值
        print(fpr)
        tpr_list.append(np.interp(mean_fpr, fpr, tpr))  # 曲线上的点坐标(fpr, tpr)
        tpr_list[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i+1, roc_auc))
        i += 1

    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tpr_list, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(tpr_list, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area={0:.2f}±{0:.2f})' .format(mean_auc, std_auc), lw=2, alpha=.8)
    # 标准差，灰色
    std_tpr = np.std(tpr_list, axis=0)
    tpr_list_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_list_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tpr_list_lower, tpr_list_upper, color='gray', alpha=.2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of '+clfStr)
    plt.legend(loc='lower right')
    plt.savefig(u'Result/'+clfStr+'.png')  # 保存图片
    plt.show()


# 保存模型
def save(clf, clfStr):
    joblib.dump(clf, 'save/'+clfStr+'.pkl')
