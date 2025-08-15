import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'Preprocessing')))

from sklearn.model_selection import train_test_split
from Preprocessing import scaling as sc  # scale_from_label, scale_from_onehot
from Preprocessing.smote import apply_smote #TEST AMAÇLI MODELİMİZDE DENEYECEGİZ KULLANIMI OPSİYONEL

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd

#xgboos modeli için verilerimiz train and test olarak ayırıyoruz
def train_and_evaluate(X, y, model_name, use_smote=False):
    X_train, X_test, y_train, y_test =train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    #smote opsiyonel 
    #öncelikle smote kullanmıyorum
    # SMOTE sadece train setine uygulanır
    #şuan kullanmamak için yorum satırına aldım 
    # if use_smote:
    #     X_train, y_train = apply_smote(X_train, y_train)

    #xcboost modelimiz
    model = XGBClassifier(
        n_estimators=200,        # Ağaç sayısı
        max_depth=4,             # Ağaç derinliği
        learning_rate=0.1,       # Öğrenme oranı
        subsample=0.8,           # Her ağaçta kullanılacak veri oranı
        colsample_bytree=0.8,    # Her ağaçta kullanılacak feature oranı
        random_state=42,
        objective="multi:softmax",  # Çok sınıflı sınıflandırma
        num_class=len(set(y))       # Sınıf sayısını otomatik algıla
    )
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)

    # Sonuçlar
    print(f"\n{model_name} Sonuçları {'(SMOTE Uygulandı)' if use_smote else ''}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Label Encoding - Normal
    X_label_scaled, y_label, _ = sc.scale_from_label(method="standard")
    
    train_and_evaluate(X_label_scaled, y_label, "Label Encoding")

    # Label Encoding - SMOTE
    use_smote = False  # Define use_smote before using it
    if use_smote == True:
    
       train_and_evaluate(X_label_scaled, y_label, "Label Encoding", use_smote=True)#somete kullanmamak için false aldım 

    # One-Hot Encoding - Normal

    X_onehot_scaled, y_onehot, _ = sc.scale_from_onehot(method="standard")

    
    train_and_evaluate(X_onehot_scaled, y_onehot, "One-Hot Encoding")


    # One-Hot Encoding - SMOTE
    if use_smote == True:

     train_and_evaluate(X_onehot_scaled, y_onehot, "One-Hot Encoding", use_smote=True)#smote kullanmamak için false aldım



    
        
