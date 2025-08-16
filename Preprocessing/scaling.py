# preprocessing adımlarından scalling ile devam ediyoruz
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Preprocessing.onehotencoding import X_encoded, y
from Preprocessing.labelencoding import df




def scale_features(X, method="standard"):
    """
    hem label hem de one hot encode için
    var olan sayısal sutünların ölçeklenmesi

     dataframe den sadece sayısal sütunlar scale edilir

    Bu işlemin amacı verileri bir noktada ölçeklemek yani :

    gaga uzunluğu 10 -20 mm arasındayken 
    kanat uzunluğu 100- 300 mm arasında olabilir
    
    modelimizin büyük değerleri baskın olarak algılamamsı için bir noktada sayısal sütunlara scale(ölçekleme) işlemi yapıyoruz

    """
    #sayısal sütunlar scale edilecek
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError("method standard veya minmax olmalı")

     # scaling ölçekleme işlemi
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X_scaled, scaler, list(X_scaled.columns)


# Label encoded veriden scale 
def scale_from_label(method="standard", target="species"):
    """
    label_encoding.py dosyasındaki df'yi alır,
    target sütununu ayırır, kalan sayısal veriyi ölçekler.
    """

# hedef değişkene scale yapmamak için dropla ayırıyorum
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    X_scaled, scaler, feature_names = scale_features(X, method=method)
    return X_scaled, y, scaler, feature_names


# One-hot encoded veriden scale 
# hedef değişkene scale uygulanmayacak zaten ayrılmıştı encode ederken
def scale_from_onehot(method="standard"):
    """
    one_hot_encoding.py dosyasındaki X_encoded ve y'yi alır,
    sayısal veriyi ölçekler.

    onehot dosyasında hedef değişkeni zaten ayırmıştık ve label encode etmiştik tekrar burda ayırmaya gerek yok
    """

    X_scaled, scaler, feature_names = scale_features(X_encoded, method=method)
    return X_scaled, y, scaler, feature_names


# Test amaçlı direkt çalıştırma
if __name__ == "__main__":
    # Label hattı
    X_label_scaled, y_label, scaler_label,feat_label = scale_from_label(method="standard")
    print("Label →", X_label_scaled.shape, len(feat_label), "features")

    # One-hot hattı
    X_onehot_scaled, y_onehot, scaler_onehot, feat_onehot = scale_from_onehot(
        method="standard")
    print("One-Hot →", X_onehot_scaled.shape, len(feat_onehot), "features")

