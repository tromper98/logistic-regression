# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:17:09 2021

@author: kolof
"""
#Подключение библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as ms

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow import keras
 
#Загрузка данных
dataset = np.loadtxt('creditcard.csv', delimiter=',', skiprows=1) #анонимизированные данные пользователей
X, y = dataset[:, 1:-1], dataset[:, -1]
#Начальное представление данных
not_fraud = sum(map(lambda x: x[-1] == 0, dataset))
fraud = sum(map(lambda x: x[-1] == 1, dataset))
print('Обычных операций', not_fraud)
print('Мошеннических операций', fraud)
#Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)


#Создание модели на Keras
model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='sigmoid', input_dim=X_train.shape[1]))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
model.summary()

#Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))


#Визуализация логистической модели
plot_model(model, show_shapes=True, show_layer_names=True)

#Предсказание результатов на основе тестовых данных
y_test_pred = model.predict_classes(X_test, batch_size=128)

# Визуализация матрицы неточностей 
labels=['Обычная', 'Мошенническая']
matrix = pd.DataFrame(confusion_matrix(y_test, y_test_pred), index=labels, columns=labels)
sns.heatmap(matrix, annot=True, cbar=None, cmap='plasma',linewidths=2,
            linecolor= 'black', fmt='g' )
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
      