# EDA ADIMINI TAMAMLADIKTAN SONRA VERİLERİ DÜZENLEYİP MODELE HAZIRLAYACAĞIMIZ PREPROCESSİNG ADIMINA GEÇİYORUZ

import pandas as pd
# label encoding için gerekli kütüphane
from sklearn.preprocessing import LabelEncoder
from Preprocessing.outlier import remove_outliers_iqr

# veriyi tekrar okuyoruz
df = pd.read_csv("palmerpenguins_extended.csv")
#iqr ile outlier temizliği yapılan veriler erişiyoruz
df = remove_outliers_iqr(df, factor=1.5)

# dublicates dropla (kopya sayısını bul ve çıkar)

before = df.shape[0]
df = df.drop_duplicates()  # drop bırak yani kopyaları sil gibi
after = df.shape[0]

# değeri 0 çıktı yani tekrar eden satır sayısı yok
if __name__ == "__main__":
    print("kopyalar silinmeden önce : ", before)
    print(" Silinen duplicate satır sayısı :", before - after)
    print("kopyalar silindikten sonra : ", after)


# kategorik olması gereken sütunları stringe çeviriyoruz
# görünüşte bu değerler zaten string gibi görünse de pandas object algılar yani her şey olabilir gibisinden bundan dolayı daha hızlı ve verimli olması için stringe dönüşümü yapılır
# ileride Label Encoding / One-Hot Encoding gibi işlemler yaparken veri tipinin net olarak string olması, beklenmeyen hataların önüne geçer.
if __name__ == "__main__":
# veri tipini görmek için örnek sonuç object olarak döner
    print(df['sex'].dtype)  


# kategorik verilerimiz ( sayısal olmayan )
categorical_cols = ["species", "island", "sex",
                    "diet", "life_stage", "health_metrics", "year"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("string")


# modelmiz için ayrıca da hangisi verimli gözlemlemek için verilerimize hem label encoding hem de one hot encoding uygulayacağız
# öncelikle label encoding
# kategorik verileri modelimizin anlayabilmesi için encode ediyoruz 
# Her kategoriye sayısal bir ID veriyoruz mesela species için (Adelie → 0, Chinstrap → 1, Gentoo → 2 gibi).
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    # modelimizi eğittikten sonra tekrar kategoriye dönüştürmek için saklıyoruz
    label_encoders[col] = le


if __name__ == "__main__":  # farklı dosyadan import edildiğinde bu verileri tekrar ekrana vermesin diye kullanıyorum
    print("label encoding tamamlandı")
    print(df.head())
