
import os
import sys
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'Preprocessing')))

from sklearn.model_selection import train_test_split
from Preprocessing import scaling as sc  # scale_from_label, scale_from_onehot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd

#xgboos modeli için verilerimiz train and test olarak ayırıyoruz
def train_and_evaluate(X, y, model_name):
    X_train, X_test, y_train, y_test =train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

 

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
    print(f"\n{model_name} Sonuçları")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    # Label Encoding - Normal
    X_label_scaled, y_label,_, feature_names_label = sc.scale_from_label(method="standard")
    
    model_label = train_and_evaluate(X_label_scaled, y_label, "Label Encoding")

    # Modeli sadece one hot encode için kaydet çünkü onun verileri daha iyi sonuç verdi tahminizi bu kaydedilen dosyadan okuyup yapacağız
    import os
    os.makedirs("saved_models", exist_ok=True)

    # One-Hot Encoding 
    X_onehot_scaled, y_onehot, _, feature_names_onehot = sc.scale_from_onehot(method="standard")
    
    model_onehot = train_and_evaluate(X_onehot_scaled, y_onehot, "One-Hot Encoding")
    model_onehot.save_model("saved_models/xgb_onehot.json")
    with open("saved_models/xgb_onehot_features.txt", "w") as f:
        f.write(",".join(feature_names_onehot))
    print(" One-Hot Encoding XGBoost modeli kaydedildi.")