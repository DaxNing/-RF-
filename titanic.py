#!/user/bin/python
# _*_ coding: UTF-8 _*_

__author__ = 'Administrator'
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("C:/Users/86135/Desktop/train.csv", dtype={"Age": np.float64},)
test = pd.read_csv("C:/Users/86135/Desktop/test.csv",dtype={"Age":np.float64})
result = pd.read_csv("C:/Users/86135/Desktop/gender_submission.csv")
#添加结果文件

def harmonize_data(titanic):
    # 填充空数据 和 把string数据转成integer表示

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = harmonize_data(train)
test_data = harmonize_data(test)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
results = []
sample_leaf_options = list(range(1, 500, 3))
n_estimators_options = list(range(1,1000, 5))
groud_truth = result['Survived'][:400]

for leaf_size in sample_leaf_options:
   for n_estimators_size in n_estimators_options:
        alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=50)
        alg.fit(train_data[predictors][:800], train_data['Survived'][:800])#输入数据训练，前者时训练数据，后者是标签
        predict = alg.predict(test_data[predictors][:400])
        # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
        results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))
        # 真实结果和预测结果进行比较，计算准确率
        print((groud_truth == predict).mean())#求取均值

# 打印精度最大的那一个三元组
print(max(results, key=lambda x: x[2]))#匿名函数 x为参数,x[2]为返回值
