# train test olacak şekilde işlenmiş verilerimizi ayırma işlemi yapıcaz model öncesi

import pandas as pd
from sklearn.model_selection import train_test_split
from scaling import scale_from_label, scale_from_onehot


# label encoded veriyi train ve test için ikiye böler
def split_label(test_size=0.2, random_state=42):
    X_label_scaled, y_label, _ = scale_from_label(method="standard")
    X_train, X_test, y_train, y_test = train_test_split(
        X_label_scaled, y_label,
        test_size=test_size,
        random_state=random_state,
        stratify=y_label
    )
    return X_train, X_test, y_train, y_test


# onehot encoded veriyi train ve test için ikiye böler
def split_onehot(test_size=0.2, random_state=42):
    X_onehot_scaled, y_onehot, _ = scale_from_onehot(method="standard")
    X_train, X_test, y_train, y_test = train_test_split(
        X_onehot_scaled, y_onehot,
        test_size=test_size,
        random_state=random_state,
        stratify=y_onehot
    )
    return X_train, X_test, y_train, y_test


# kodu derlediğimizde sonucun
if __name__ == "__main__":
    # Test için çalıştırma
    X_train_l, X_test_l, y_train_l, y_test_l = split_label()
    print(f"Label → Train: {X_train_l.shape}, Test: {X_test_l.shape}")

    X_train_o, X_test_o, y_train_o, y_test_o = split_onehot()
    print(f"One-Hot → Train: {X_train_o.shape}, Test: {X_test_o.shape}")
