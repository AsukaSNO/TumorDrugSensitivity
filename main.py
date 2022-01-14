import data
from model import *
import numpy as np
import pandas as pd
import joblib


# 应用，首先要规范化数据 //test
def use():
    print("以列表形式输入基因组值\n")
    temp_list = []
    while True:
        try:
            temp_list.append(input(''))
        except:
            break
    for i in temp_list:
        x = np.matrix(i.split("\t")[1:-1])
        X = pd.DataFrame(x)
        clf = joblib.load('save/clf_test.pkl')
        y = clf.predict(X)
        print(i, "\n")
        print(y, "\n\n\n\n")


if __name__ == '__main__':
    # clist = [boostSVM, stack2]
    # clistStr = ["boostSVM", "stackList2"]
    # rst_data = get_score(clist, clistStr)
    # write_to_file(rst_data)
    draw_roc(SVM, "SVM1", X, y)
    # draw_roc(stack2, "stack2", X, y)


