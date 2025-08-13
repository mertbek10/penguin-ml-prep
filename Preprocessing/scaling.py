# preprocessing adımlarından scalling ile devam ediyoruz
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_features(X, method="standard"):
    """
    hem label hem de one hot encode için
    var olan sayısal sutünların ölçeklenmesi

    X dataframe den sadece sayısal sütunlar scale edilir

    method: standar std veya minmax 

    Dönüş:
    scaled_X: DataFrame -> Ölçeklenmiş veriler
    scaler: fit edilmiş scaler nesnesi

    """

    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError("method standard veya minmax olmalı")

     # Ölçekleme işlemi
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X_scaled, scaler


# --- Label encoded veriden scale ---
def scale_from_label(method="standard", target="species"):
    """
    label_encoding.py dosyasındaki df'yi alır,
    target sütununu ayırır, kalan sayısal veriyi ölçekler.
    """
    from labelencoding import df

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_scaled, scaler = scale_features(X, method=method)
    return X_scaled, y, scaler


# --- One-hot encoded veriden scale ---
def scale_from_onehot(method="standard"):
    """
    one_hot_encoding.py dosyasındaki X_encoded ve y'yi alır,
    sayısal veriyi ölçekler.
    """
    from onehotencoding import X_encoded, y

    X_scaled, scaler = scale_features(X_encoded, method=method)
    return X_scaled, y, scaler


# Test amaçlı direkt çalıştırma
if __name__ == "__main__":
    # Label hattı
    X_label_scaled, y_label, scaler_label = scale_from_label(method="standard")
    print("Label →", X_label_scaled.shape)

    # One-hot hattı
    X_onehot_scaled, y_onehot, scaler_onehot = scale_from_onehot(
        method="standard")
    print("One-Hot →", X_onehot_scaled.shape)
    import onehotencoding
