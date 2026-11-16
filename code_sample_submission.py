import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# read CSV

file_path_train = 'D:\\Books\\project\\0001-spaceship-titanic\\train.csv'
dataset = pd.read_csv(file_path_train)

# create good data and fill NaN (train)

dataset[['Deck', 'CabinNum', 'Side']] = dataset['Cabin'].str.split('/', expand=True)

for i in dataset[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]:
    dataset[i]=dataset[i].fillna(0)

dataset['pay']=dataset['RoomService']+dataset['FoodCourt']+dataset['ShoppingMall']+dataset['Spa']+dataset['VRDeck']
dataset['CryoSleep']=dataset['CryoSleep'].fillna(dataset['pay'].apply(lambda x: True if x==0 else False))
dataset['CryoSleep']=dataset['CryoSleep'].replace({True : 1 , False : 0})

dataset['Deck'] = dataset['Deck'].fillna(dataset['Deck'].mode()[0])
dataset['Deck'] = dataset['Deck'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})

dataset['Side'] = dataset['Side'].fillna(dataset['Side'].mode()[0])
dataset['Side'] = dataset['Side'].replace({'S':1 , 'P':0})

dataset['Age']=dataset['Age'].fillna(dataset['Age'].mean())

dataset['VIP'] = dataset['VIP'].fillna(False).replace({True: 1, False: 0})

dataset['Destination']=dataset['Destination'].fillna('Unknown').replace({'TRAPPIST-1e': 0,'55 Cancri e': 1,'PSO J318.5-22': 2,'Unknown': 3})

dataset['Transported']=dataset['Transported'].replace({True:1 , False:0}).astype(int)

dataset[['CryoSleep','Age','VIP','Deck','Side','Destination']]=dataset[['CryoSleep','Age','VIP','Deck','Side','Destination']].astype(float)

# create and fit Model , train and test

X_feature = np.array(['CryoSleep','Age','VIP','Deck','Side','Destination','pay'])

X = dataset[X_feature]
y = dataset['Transported']

x_train , x_test , y_train , y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = xgboost.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
model.fit(x_train, y_train)

# Test

file_path_test = 'D:\\Books\\project\\0001-spaceship-titanic\\test.csv'
dataset_test = pd.read_csv(file_path_test)

passenger_ids = dataset_test['PassengerId']

dataset_test[['Deck', 'CabinNum', 'Side']] = dataset_test['Cabin'].str.split('/', expand=True)

for i in dataset_test[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']]:
    dataset_test[i] = dataset_test[i].fillna(0)

dataset_test['pay'] = dataset_test['RoomService'] + dataset_test['FoodCourt'] + dataset_test['ShoppingMall'] + dataset_test['Spa'] + dataset_test['VRDeck']
dataset_test['CryoSleep'] = dataset_test['CryoSleep'].fillna(dataset_test['pay'].apply(lambda x: True if x == 0 else False))
dataset_test['CryoSleep'] = dataset_test['CryoSleep'].replace({True: 1, False: 0})

dataset_test['Deck'] = dataset_test['Deck'].fillna(dataset_test['Deck'].mode()[0])
dataset_test['Deck'] = dataset_test['Deck'].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8})

dataset_test['Side'] = dataset_test['Side'].fillna(dataset_test['Side'].mode()[0])
dataset_test['Side'] = dataset_test['Side'].replace({'S': 1, 'P': 0})

dataset_test['Age'] = dataset_test['Age'].fillna(dataset_test['Age'].mean())

dataset_test['VIP'] = dataset_test['VIP'].fillna(False).replace({True: 1, False: 0})

dataset_test['Destination'] = dataset_test['Destination'].fillna('Unknown').replace({
    'TRAPPIST-1e': 0,
    '55 Cancri e': 1,
    'PSO J318.5-22': 2,
    'Unknown': 3
})

dataset_test[['CryoSleep', 'Age', 'VIP', 'Deck', 'Side', 'Destination']] = dataset_test[['CryoSleep', 'Age', 'VIP', 'Deck', 'Side', 'Destination']].astype(float)

X_feature_test = np.array(['CryoSleep', 'Age', 'VIP', 'Deck', 'Side', 'Destination', 'pay'])

X_test = dataset_test[X_feature_test]

predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': passenger_ids,
    'Transported': predictions.astype(bool)
})

submission.to_csv('D:\\Books\\project\\0001-spaceship-titanic\\FF.csv',index=False)