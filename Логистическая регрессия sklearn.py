# -*- coding: utf-8 -*-
"""
Created on Tue Jcn  5 13:28:51 2021

@author: kolof
"""

#Подключение библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as ms
import seaborn as sns

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Загрузка данных
dataset = np.loadtxt('creditcard.csv', delimiter=',', skiprows=1) #анонимизированные данные пользователей
X, y = dataset[:, 1:-1], dataset[:, -1]

#Начальное представление данных
not_fraud = sum(map(lambda x: x[-1] == 0, dataset))
fraud = sum(map(lambda x: x[-1] == 1, dataset))
print('Обычных операций', not_fraud)
print('Мошеннических операций', fraud)

#Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)

#Создание линейного регрессора
Linear_regressor = linear_model.LogisticRegression()
#Обучение линейного регрессора
Linear_regressor.fit(X_train, y_train)
#Предсказание данных на основе тестовых входных данных
y_test_pred = Linear_regressor.predict(X_test)
# Визуализация матрицы ошибок
labels=['Обычная', 'Мошенническая']
matrix = pd.DataFrame(confusion_matrix(y_test, y_test_pred), index=labels, columns=labels)
sns.heatmap(matrix, annot=True, cbar=None, cmap='plasma',linewidths=2,
            linecolor= 'black', fmt='g')
plt.title('Матрица ошибок')
plt.tight_layout()
plt.ylabel('Истинные исходы')
plt.xlabel('Предсказанные исходы')
plt.show()

#Отражение статистики алгоритма
print('Accuracy:', round(ms.accuracy_score(y_test, y_test_pred), 5))
print('Precision:', round(ms.precision_score(y_test, y_test_pred), 5))
print('Recall:', round(ms.recall_score(y_test, y_test_pred),4))
print('F1-score:', round(ms.f1_score(y_test, y_test_pred),4))

print(ms.classification_report(y_test, y_test_pred, target_names=labels))
      