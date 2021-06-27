import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib as mpl

####################
######Görev 1#######
####################


########: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz
df = pd.read_csv("persona.csv")

df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

####  Kaç unique SOURCE vardır? Frekansları nedir?

print(df["SOURCE"].nunique())
print(df["SOURCE"].value_counts())

""""
2
android    2974
ios        2026
Name: SOURCE, dtype: int64

"""
# Soru 3: Kaç unique PRICE vardır?

print(df["PRICE"].nunique())
print(df["PRICE"].value_counts())

"""
6
29    1305
39    1260
49    1031
19     992
59     212
9      200
Name: PRICE, dtype: int64

"""
#Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

print(df["PRICE"].value_counts())

"""
29    1305
39    1260
49    1031
19     992
59     212
9      200
"""

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

print(df["COUNTRY"].value_counts())
"""
usa    2065
bra    1496
deu     455
tur     451
fra     303
can     230
"""

#Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

print(df[["COUNTRY", "PRICE"]].groupby("COUNTRY").agg({"sum"}))

"""
PRICE
           sum
COUNTRY       
bra      51354
can       7730
deu      15485
fra      10177
tur      15689
usa      70225
"""


# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?

print(df[["COUNTRY", "PRICE"]].groupby("COUNTRY").agg({"count"}))
"""
       PRICE
        count
COUNTRY      
bra      1496
can       230
deu       455
fra       303
tur       451
usa      2065
"""

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

print(df[["COUNTRY", "PRICE"]].groupby("COUNTRY").agg({"mean"}))
"""
           PRICE
              mean
COUNTRY           
bra      34.327540
can      33.608696
deu      34.032967
fra      33.587459
tur      34.787140
usa      34.007264

"""

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

print(df[["SOURCE", "PRICE"]].groupby("SOURCE").agg({"mean"}))
"""
       PRICE
              mean
SOURCE            
android  34.174849
ios      34.069102

"""

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?

print(df[["SOURCE","COUNTRY", "PRICE"]].groupby(["SOURCE","COUNTRY"]).agg(["mean"]))

"""
PRICE
                      mean
SOURCE  COUNTRY           
android bra      34.387029
        can      33.330709
        deu      33.869888
        fra      34.312500
        tur      36.229437
        usa      33.760357
ios     bra      34.222222
        can      33.951456
        deu      34.268817
        fra      32.776224
        tur      33.272727
        usa      34.371703
"""


######Görev 2#######

### COUNTRY, SOURCE, SEX, AGE kırılımında toplam kazançlar nedir?

ek = ["COUNTRY","SOURCE","SEX","AGE"]
df.groupby(ek).agg({"PRICE": "sum"})
"""
 SOURCE  ...    AGE
                                                     sum  ...    sum
PRICE                                                     ...       
9      androidiosiosandroidandroidandroidiosiosandroi...  ...   4638
19     androidandroidandroidiosandroidandroidandroida...  ...  23659
29     androidandroidandroidandroidiosandroidandroida...  ...  30643
39     androidandroidandroidandroidandroidandroidandr...  ...  29894
49     androidandroidandroidandroidandroidandroidiosi...  ...  24240
59     androidandroidandroidiosandroidandroidandroidi...  ...   4833

"""


######Görev 3#######


### Çıktıyı PRICE’a göre sıralayınız.

agg_df = df.groupby(ek).agg({"PRICE": "sum"}).sort_values("PRICE",ascending= False)
agg_df


######Görev 4#######


###Index’te yer alan isimleri değişken ismine çeviriniz.

agg_df = agg_df.reset_index()


######Görev 5#######


# age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0,18,24,30,40,70], labels=["0_18","19_23","24_30","31_40","41_70"])
agg_df


######Görev 6#######
###########Yeni seviye tabanlı müşterileri (persona) tanımlayınız.

agg_df["CUSTOMER_LEVEL_BASED"] = [str(agg_df["COUNTRY"][i]).upper() + "_" + str(agg_df["SOURCE"][i]).upper() + "_" + str(agg_df["SEX"][i]).upper() + "_" + str(agg_df["AGE_CAT"][i]).upper()  for i in range(agg_df.shape[0])]

agg_df = agg_df[["CUSTOMER_LEVEL_BASED","PRICE"]]

agg_df = agg_df.groupby("CUSTOMER_LEVEL_BASED").agg({"PRICE":"mean"}).reset_index()
agg_df

######Görev 7#######

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels = ["D","C","B","A"])

agg_df.groupby("SEGMENT").agg(["mean","max","sum"]).sort_values("SEGMENT", ascending= False)
agg_df

print(agg_df[agg_df["SEGMENT"] == "C" ].describe)
print(agg_df[agg_df["SEGMENT"] == "C"].isnull().sum())

# Yeni gelen müşterileri segmentlerine göre sınıflandırınız ve ne kadar gelir getirebileceğini tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
        # TUR_ANDROID_FEMALE_31_40
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
        # FRA_IOS_FEMALE_31_40
for i in range(3):
    users = input("please enter the information from user:")
    information = [agg_df[agg_df["CUSTOMER_LEVEL_BASED"] == str(users)]]
    print(information)
