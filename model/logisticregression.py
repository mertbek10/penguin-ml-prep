# verilerimizi modele hazırladık logisticregression ile makine öğrenimine başlayıp sonuçları değerlendiricez

import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__),  'Preprocessing')))
from sklearn.model_selection import train_test_split
from Preprocessing import scaling as sc  # scale_from_label, scale_from_onehot
#from Preprocessing.smote import apply_smote
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression
import pandas as pd


#verielrimizi train test olarak yüzde 20 test yüzde 80 train olarak ayırdık 
def train_and_evaluate(X, y, model_name, use_smote=False):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE sadece train setine uygulanır
    # if use_smote:
    #     smote = ADASYN(random_state=42)
    #     X_train = X_train.astype(float)
    #     y_train = y_train.astype(float)

    #     X_train, y_train = smote.fit_resample(X_train, y_train)
    #     print(f"SMOTE sonrası sınıf dağılımı: {pd.Series(y_train).value_counts().to_dict()}")


    # Model oluştur
    model = LogisticRegression(max_iter=1000)#mac iter testi güncelleme sınırı 
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)

    # Sonuçlar
    
    print(f"\n{model_name} Sonuçları {'(ADASYN Uygulandı)' if use_smote else ''}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Label Encoding - Normal
    X_label_scaled, y_label, _ = sc.scale_from_label(method="standard")
    
    train_and_evaluate(X_label_scaled, y_label, "Label Encoding")

    # Label Encoding - SMOTE
    # train_and_evaluate(X_label_scaled, y_label, "Label Encoding", use_smote=True)

    # One-Hot Encoding - Normal

    X_onehot_scaled, y_onehot, _ = sc.scale_from_onehot(method="standard")
    train_and_evaluate(X_onehot_scaled, y_onehot, "One-Hot Encoding")


    # One-Hot Encoding - SMOTE
    # train_and_evaluate(X_onehot_scaled, y_onehot, "One-Hot Encoding", use_smote=True)

