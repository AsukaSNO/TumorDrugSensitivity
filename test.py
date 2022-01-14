# 没用，随便测试点啥
from numpy import *
import matplotlib.pyplot as plt
from model import *
from data import *
from sklearn.metrics import accuracy_score
def loadData():
    dataMat = matrix([[1. , 2.1],
                      [2. , 1.1],
                      [1.3, 1. ],
                      [1. , 1. ],
                      [2. , 1. ]])
    classLabels = [1, 1, -1, -1, 1]
    return dataMat, classLabels

X, y = read_csv()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
#
def stumpClassify(X, dimen, threshVal, threshIneq):
    retArray = ones((shape(X)[0], 1))
    if threshIneq == 'lt':
        retArray[X[:, dimen] <= threshVal] = 0
    else:
        retArray[X[:, dimen] > threshVal] = 0
    return retArray


def buildStump(X, y, D):
    X = mat(X)
    labelMat = mat(y).T  # 转置
    m, n = shape(X)  # 样本数，维度
    numSteps = 10.0
    bestStump = {}  # 最好树桩（单层决策树）
    bestClassEst = mat(zeros((m, 1)))
    minError = inf  # 最小错误率设为+inf
    for i in range(n):  # 对数据集中的每一个特征
        rangeMin = X[:, i].min()  # 每个特征的最小值
        rangeMax = X[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps  # 步长，分成10步
        for j in range(-1, int(numSteps)+1):  # 对每个步长
            for inequal in ['lt', 'gt']:  # 对每个不等号
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(X, i, threshVal, inequal)  # 当前预测
                errArr = mat(ones((m, 1)))  # 初始化1
                errArr[predictedVals == labelMat] = 0  # 没错的置为False
                weightedError = D.T * errArr  # 加权错误率
                # print("dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %\
                #       (i, threshVal, inequal, weightedError))
                if(weightedError < minError):
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i  # 第dim个特征,从0开始
                    bestStump['thresh'] = threshVal  # 当前特征值
                    bestStump['ineq'] = inequal  # 和阈值相比
    return bestStump, minError, bestClassEst


def adaBoostTrain(X, y, numIt = 40):
    weakClassArr = []
    m = shape(X)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(X, y, D)
        # print("D:", D.T)
        alpha = float(0.5*log((1-error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print("classEst", classEst.T)
        expon = multiply(-1*alpha*mat(y).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        # print("aggClassEst:", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(y).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        # print("total error", errorRate, "\n")
        if errorRate == 0.0:
            break
        print(i, "t")
    return weakClassArr



def adaClassify(X, classifierArr):
    dataMatrix = mat(X)
    m = shape(X)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(X, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return sign(aggClassEst)

D = mat(ones((231, 1)) / 231)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
classifierArr = adaBoostTrain(X_train, y_train, numIt=10)
y_rst = adaClassify(X_train, classifierArr)
print(shape(y_rst))
print(accuracy_score(y_rst, y_train))
