import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import random
import numpy as np
from sklearn import preprocessing
import sklearn.metrics as metrics
import pickle
from sklearn.model_selection import train_test_split

!wget https://github.com/Jshondra/dataset/blob/main/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.zip?raw=true -O dataset.zip
!unzip -u dataset.zip
df = pd.read_csv('Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', engine='python')
df.shape
df.columns = df.columns.str.strip()
df = df.drop(columns=['Fwd Header Length.1'])
df.shape
df['Label'].unique()
df['Label'].value_counts()
df = df.drop(df[pd.isnull(df['Flow ID'])].index)
df.shape
df.replace('Infinity', -1, inplace=True)
df[["Flow Bytes/s", "Flow Packets/s"]] = df[["Flow Bytes/s", "Flow Packets/s"]].apply(pd.to_numeric)
df.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
string_features = list(df.select_dtypes(include=['object']).columns)
string_features.remove('Label')
string_features
le = preprocessing.LabelEncoder()
df[string_features] = df[string_features].apply(lambda col: le.fit_transform(col))
benign_total = len(df[df['Label'] == "BENIGN"])
benign_total
attack_total = len(df[df['Label'] != "BENIGN"])
attack_total
df.to_csv("web_attacks_unbalanced.csv", index=False)
df['Label'].value_counts()
enlargement = 1.1
benign_included_max = attack_total / 30 * 70
benign_inc_probability = (benign_included_max / benign_total) * enlargement
print(benign_included_max, benign_inc_probability)
indexes = []
benign_included_count = 0
for index, row in df.iterrows():
    if (row['Label'] != "BENIGN"):
        indexes.append(index)
    else:
        # Copying with benign_inc_probability
        if random.random() > benign_inc_probability: continue
        # 70%
if benign_included_count > benign_included_max: continue
        benign_included_count += 1
        indexes.append(index)
df_balanced = df.loc[indexes]
df_balanced = df_balanced[df_balanced['Label'] != "Web Attack – Brute Force"]
df_balanced = df_balanced[df_balanced['Label'] != "Web Attack – Sql Injection"]
df_balanced['Label'].value_counts()
df_balanced.to_csv("web_attacks_balanced.csv", index=False)
df = pd.read_csv('web_attacks_balanced.csv')
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
y = df['Label'].values
webattack_features = ['Average Packet Size','Flow Bytes/s','Max Packet Length','Fwd Packet Length Mean','Fwd IAT Min','Total Length of Fwd Packets',
                      'Flow IAT Mean','Fwd Packet Length Max','Fwd IAT Std','Fwd Header Length']
X = df[webattack_features]
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))
df = pd.read_csv('web_attacks_balanced.csv')
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
excluded = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp']
df = df.drop(columns=excluded, errors='ignore')
excluded2 = ['Init_Win_bytes_backward', 'Init_Win_bytes_forward']
df = df.drop(columns=excluded2, errors='ignore')
y = df['Label'].values
X = df.drop(columns=['Label'])
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X[webattack_features], y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
rfc = RandomForestClassifier(max_depth=17, max_features=10, min_samples_leaf=3, n_estimators=50, random_state=42, oob_score=True)
# rfc = RandomForestClassifier(n_estimators=250, random_state=1)
rfc.fit(X_train, y_train)
features = X.columns
importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]
for index, i in enumerate(indices[:10]):
    print('{}.\t#{}\t{:.3f}\t{}'.format(index + 1, i, importances[i], features[i]))
y_pred = rfc.predict(X_test)
confusion_matrix(y_test, y_pred)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
print('Accuracy =', accuracy)
print('Precision =', precision)
print('Recall =', recall)
print('F1 =', f1)
with open('webattack_detection_rf_model.pkl', 'wb') as f:
    pickle.dump(rfc, f)
seconds = time.time()
y_pred = rfc.predict(X_test)
print("Total operation time:", time.time() - seconds, "seconds")
print("Benign records detected (0), attacks detected (1):")
unique, counts = np.unique(y_pred, return_counts=True)
dict(zip(unique, counts))
