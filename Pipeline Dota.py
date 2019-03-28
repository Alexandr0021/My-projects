import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
import time
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

#функция нахождения пустых столбцов
def check_nan(data):
    a = data.isnull().any()
    k = 0
    array = []
    for i in a:
        if i == True:
            array.append(data.columns[k])
        k = k + 1
    return array
#заполняем пустые значения средними 
def fillnan(data):
    for i in check_nan(data):
        for j in data[i]:
            avg = 0
            avg = avg + j
        avg = avg/(len(data[i]))
        data[i] = data[i].fillna(avg)
    return data

NaN_train = check_nan(data_train)
data_train = fillnan(data_train)
data_train['dire_flying_courier_time'] = data_train['dire_flying_courier_time'].fillna(600)

#целевая переменная - radiant_win
scaler = MinMaxScaler()
y = data_train['radiant_win']
X = data_train.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1)

X = scaler.fit_transform(X)

scores = []
kf = KFold(y.size, n_folds=5, shuffle=True, random_state=1)
for i in (10, 20, 30):
    start_time = datetime.datetime.now()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = GradientBoostingClassifier(n_estimators = i)
    #clf.fit(X, y)
    
    scores.append(np.mean(cross_val_score(clf, X, y, cv=kf, scoring = 'roc_auc')))
    #pred = clf.predict_proba(X_test)[:, 1]
    print ('Time elapsed:', datetime.datetime.now() - start_time)
print(scores)

#Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? 
#Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?
for i in (40, 50):
    scores = []
    start_time = datetime.datetime.now()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = GradientBoostingClassifier(n_estimators = i)
    #clf.fit(X, y)
    
    scores.append(np.mean(cross_val_score(clf, X, y, cv=kf, scoring = 'roc_auc')))
    #pred = clf.predict_proba(X_test)[:, 1]
    print(scores)
    print ('Time elapsed:', datetime.datetime.now() - start_time)
#смысл есть, но маленький (выгода небольшая)
    
#Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? 
#Чем вы можете объяснить это изменение?
X = X.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero'
           , 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero', 'match_id', 'start_time'], axis=1)
scores = []
#start_time = datetime.datetime.now()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(penalty='l2', C = 2.7)
clf.fit(X, y)

scores.append(np.mean(cross_val_score(clf, X, y, cv=kf)))
        #pred = clf.predict_proba(X_test)[:, 1]
print(scores)
#результат 0.65, что не может не радовать. Логистическая регрессия хорошо строится на численных данных без значений, которым соответствует что-то, кроме чисел.
#например, start_time - бесполезная переменная, которая говорит только о начале матча. На исход никак не влияет.

#Какое получилось качество при добавлении "мешка слов" по героям? 
#Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?
# N — количество различных героев в выборке
X_pick = np.zeros((data_train.shape[0], max(np.unique(a))))

for i, match_id in enumerate(data_train.index):
    for p in range(5):
        X_pick[i, data_train.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data_train.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
        
X_pick = pd.DataFrame(X_pick)
X_new = pd.merge(X, X_pick, on=X.index)

#Выбор модели 1
scores = []
#start_time = datetime.datetime.now()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LogisticRegression(penalty='l2', C = 1.5)
clf.fit(X_new, y)

scores.append(np.mean(cross_val_score(clf, X_new, y, cv=kf, scoring = 'roc_auc')))
        #pred = clf.predict_proba(X_test)[:, 1]
print(scores)

#2
scores = []
#start_time = datetime.datetime.now()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = GradientBoostingClassifier(n_estimators = 50)
clf.fit(X_new, y)

scores.append(np.mean(cross_val_score(clf, X_new, y, cv=kf, scoring = 'roc_auc')))
        #pred = clf.predict_proba(X_test)[:, 1]
print(scores)

data_test = fillnan(data_test)
check_nan(data_test)

X_test = data_test.drop(['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero'
           , 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero', 'match_id', 'start_time'], axis=1)
X_test = scaler.fit_transform(X_test)    
a = np.array(data_test[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero'
           , 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']])
X_pick_test = np.zeros((data_test.shape[0], max(np.unique(a))))

for i, match_id in enumerate(data_test.index):
    for p in range(5):
        X_pick_test[i, data_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, data_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
        
X_pick_test = pd.DataFrame(X_pick_test)
X_test_new = pd.merge(X_test, X_pick_test, on=X_test.index)

#Убедитесь, что предсказанные вероятности адекватные — находятся на отрезке [0, 1], не совпадают между собой (т.е. что модель не получилась константной).
ids = data_test['match_id']
predictions = gbc.predict_proba(X_test_new)
#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'match_id' : ids, 'radiant_win': predictions })
output.to_csv('Dota2_submit_coursera.csv', index=False)