import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# 讀取 CSV 檔案
train_data = pd.read_csv('resource/train/SNR2_L8/L=8_frame=100000_SNR=2.csv')  # 100,000 samples
test_data = pd.read_csv('resource/test/SNR2_L8/L=8_frame=25000_SNR=2.csv')    # 10,000 samples


# 假設資料中的最後一欄是標籤 (target)，其餘為特徵
X_train = train_data.iloc[:, :-1]  # 前 8 個欄位是特徵
y_train = train_data.iloc[:, -1]   # 最後一個欄位是標籤

X_test = test_data.iloc[:, :-1]    # 前 8 個欄位是特徵
y_test = test_data.iloc[:, -1]     # 最後一個欄位是標籤


# 初始化隨機森林模型
rf = RandomForestClassifier(n_estimators=500, random_state=42)

# 訓練模型
rf.fit(X_train, y_train)

# 預測
y_pred = rf.predict(X_test)

# 輸出準確率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Precision 和 Recall
report = classification_report(y_test, y_pred)
print(report)