'''
Tahminlerimizi input olarak terminalden gireceğiz

modeli eğittik kaydettik şimdi test etme sırasında 

'''

import pandas as pd
from xgboost import XGBClassifier

#modeli yüklüyoruz 
model = XGBClassifier()
model.load_model("saved_models/xgb_onehot.json")

#feature isimlerini oku
with open("saved_models/xgb_onehot_features.txt", "r") as f:
    feature_names = f.read().strip().split(",")

print(f"Model yüklendi ({len(feature_names)} feature var)")

#tahminler için input alma
input_values =[]
for feature in feature_names:
    val = input(f"{feature} değerini gir : ")
    try :
        #sayısal featureleri floata çevir
        val = float(val)

    except ValueError:
        #buraya dönüş yapılcak 
        pass
    input_values.append(val)


# DataFrame oluştur
user_input = pd.DataFrame([input_values], columns=feature_names)
mapping = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

# Tahmin yap
prediction = model.predict(user_input)[0]
print("Tahmin : ", mapping.get(prediction, "Bilinmeyen sınıf"))

