# verilerimizi modele hazırladık logisticregression ile makine öğrenimine başlayıp sonuçları değerlendiricez
from sklearn.model_selection import train_test_split
from Preprocessing.scaling import scale_from_label, scale_from_onehot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd


def train_and_evaluate(X, y, model_name):
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Model oluştur
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)

    # Sonuçlar
    print(f"\n {model_name} Sonuçları")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    # Label Encoding
    X_label_scaled, y_label, _ = scale_from_label(method="standard")
    train_and_evaluate(X_label_scaled, y_label, "Label Encoding")

    # One-Hot Encoding
    X_onehot_scaled, y_onehot, _ = scale_from_onehot(method="standard")
    train_and_evaluate(X_onehot_scaled, y_onehot, "One-Hot Encoding")
