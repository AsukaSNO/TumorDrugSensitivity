import pandas as pd
# interface
drugID = "1005"  # 药品名
source = "C:/Users/Machenike/Desktop/TumorDrugSensitivity"  # 工程路径


# 读取.csv
def read_csv(path=u"Drugs/DNormJW "+drugID+" .csv"):
    matrix = pd.read_csv(path)
    y = matrix['IC_class']
    X = matrix.drop(columns=['Unnamed: 0', 'IC_class', 'LN_IC50', 'IC_norm2'])
    return X, y
