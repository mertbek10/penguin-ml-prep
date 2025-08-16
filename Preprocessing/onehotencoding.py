# one hot encoding için ayrı bir dosyada çalışıyorum farklı modeller için hem label hem one hot encoding kullanıcam o yüzden dosyayı tekrar okudum
import pandas as pd
# sadece hedef değişken için label encode yapılacak again
from sklearn.preprocessing import LabelEncoder
from Preprocessing.outlier import remove_outliers_iqr


df = pd.read_csv("palmerpenguins_extended.csv")
df = remove_outliers_iqr(df, factor=1.5)


categorical_cols = ["island", "sex",
                    "diet", "life_stage", "health_metrics", "year"]

# Hedef değişkeni label encode et
le_target = LabelEncoder()
y = le_target.fit_transform(df["species"].astype("string"))

# eksik verileri label da kontrol ettiğim için tekrar bakmıyorum zaten eksik veri yok
# veya duplicates etmeme gerek yok

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("string")


# hedef değişkeni çıkardık
X = df.drop(columns=["species"]).copy()

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

if __name__ == "__main__":
    print("One-Hot Encoding tamamlandı!")
    print(X_encoded.head())


# Encode edilmiş hedef değişkenin geri dönüş map'i
target_mapping = dict(
    zip(le_target.classes_, le_target.transform(le_target.classes_)))
if __name__ == "__main__":
    print("Hedef değişken mapping:", target_mapping)
