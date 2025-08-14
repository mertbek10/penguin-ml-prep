# modelden sonra eksik veride çoğaltma kararı aldık (oversamplin) ve smoote tekniği uygulayacağız
from imblearn.over_sampling import ADASYN

import pandas as pd

def apply_smote(X_train, y_train, k_neighbors=5, random_state=42):
    """
    Sadece eğitim verisine SMOTE uygular.
    Test setine kesinlikle dokunmaz.
    veriler arasındaki komşuları bularak buralardan referans alarak benzer kopyalar oluşturur
    rastgele komşular değil

        X_train ( Eğitim özellikleri
        y_train  Eğitim etiketleri
        k_neighbors  SMOTE komşu sayısı
        random_state  Rastgelelik kontrolü

    """
    print(" SMOTE öncesi sınıf dağılımı : ")
    print(pd.Series(y_train).value_counts())

    sm = ADASYN(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("\n SMOTE sonrası sınıf dağılımı : ")
    print(pd.Series(y_res).value_counts())

    return X_res, y_res
