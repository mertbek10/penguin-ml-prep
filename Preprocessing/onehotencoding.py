# one hot encoding için ayrı bir dosyada çalışıyorum
#  modeller sonuçlarını karşılaştırırken onehot ve label ile encode edilmiş verilerin çıktısını karşılaştırmak için
import pandas as pd
# sadece hedef değişken için label encode yapılacak 
from sklearn.preprocessing import LabelEncoder
from Preprocessing.outlier import remove_outliers_iqr

'''
çoğu işlem label dakiyle aynı

- dosyayı tekrar oku
- kategorik sütunların tipini stringe dönüştür


farklı olark hedef değişkene label encode edilir 
Bunun sebebi label encode mantığı şu şekildedir :

adeli 0 gento 1 chinstrap 2
biz hedef değişkeni bu şekilde sınıflandırmak isteriz (target --> label encode)

one hot encode da ise :
       adeli 1 gentoo 0 chistrap 0
       adeli 0 gentoo 1  chistrap 0
       adeli 0 gentoo 0  chistrap 1
       
gibi matrix olarak dönüşüm yapar 
one hot encode işleminde sütun sayısı artar
verilerin encode edilmiş hallerinin model sonuçlarına etkisini inceleyip iyi sonuç veren encode yöntemiyle test sonuçlarına geçilecek
'''

df = pd.read_csv("palmerpenguins_extended.csv")
#uç değer temizliği yapılmış verilere erişiyoruz 
df = remove_outliers_iqr(df, factor=1.5)


categorical_cols = ["island", "sex",
                    "diet", "life_stage", "health_metrics", "year"]

# Hedef değişkeni label encode et
le_target = LabelEncoder()
y = le_target.fit_transform(df["species"].astype("string"))


for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("string")


# hedef değişkeni çıkardık
# scale işlemine hedef değişken sokulmayacak
X = df.drop(columns=["species"]).copy()

#encode işlemi 
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# Ada sütunlarının ağırlığını azalt (ör. 0.8 ile çarp)
#model öğrenirken ada sütununu baskın almasın diye ağırlıığını azaltıyoruz 
island_cols = [col for col in X_encoded.columns if col.startswith("island_")]
X_encoded[island_cols] = X_encoded[island_cols] * 0.9

if __name__ == "__main__":
    print("One-Hot Encoding tamamlandı!")
    print(X_encoded.head())


# Encode edilmiş hedef değişkenin geri dönüş map'i
target_mapping = dict(
    zip(le_target.classes_, le_target.transform(le_target.classes_)))
if __name__ == "__main__":
    print("Hedef değişken mapping:", target_mapping)

