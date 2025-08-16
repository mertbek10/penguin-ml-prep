
# model değerlendirmeleri sonucu doğruluk oranını artırmak amacıyla outlier temizliği yapmaya karar verdik 
# EDA adımlarında outlierlerin varlığını ıqr ile zaten test etmiştik şimdi model sonuçları sonrası daha temiz veri almak için outlier işlemi yapılcak
import pandas as pd

def remove_outliers_iqr(df, columns=None, factor=1.5):

    df_clean = df.copy()

    # Kolonlar belirlenmemişse numeric olanları seç
    if columns is None:
        columns = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Outlier olmayan satırları filtrele
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean
