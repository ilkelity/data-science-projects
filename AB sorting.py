import pandas as pd
import math
import scipy.stats as st
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

from scipy.stats import shapiro
import scipy.stats as stats

df_control = pd.read_excel(r"5.hafta/ilke akar - 5.hafta/ab_testing.xlsx" , sheet_name="Control Group")
df_test = pd.read_excel(r"5.hafta/ilke akar - 5.hafta/ab_testing.xlsx" , sheet_name="Test Group")

## normallik varsayımı , normal dağılımı inceleriz, shapiro kullanırız
# h0 için normallik sağlanmaktadır
# h1 ... sağlanmamakta

test_ıstatıstık, p_value = shapiro(df_control["Purchase"])
print(p_value) ## h0 red edilemedi


test_ıstatıstık, p_value = shapiro(df_test["Purchase"])
print(p_value) ## h0 red edilemedi

## hem test grubu hem de

## varyans homojenliğinde de levene kullanıyoruz

# h0 = VARYANSLAR homojendir
# h1 .. değildir

test_ıstatıstık, p_value = stats.levene(df_control["Purchase"], df_test["Purchase"])
print(p_value) ## h0 red edilemedi

## normallik varsayımı ve varyans homojenliği sağlandığı için bağımsız iki örneklem T testi kullanılır
## bagımsız gruplar : max vs average

test_ıstatıstık, p_value = stats.ttest_ind(df_control["Purchase"], df_test["Purchase"])
print(p_value) ## h0 red edilemedi

## Test ve kontrol grupları arasında anlamlı bir fark yoktur

## ADIM 3: hangi testi kullandınız ve neden?

# normallik varsayımı için shapiro kullanıldı,
# iki grupta da Normal dağılım olduğu için varyans analizine geçtik
# varyans homojenliği levene testini kullandım, varyanslar da homojen old için
# sonuç olarak Bagımsız iki örneklem T testi kullanılır

## Görev 4: Görev 2’de verdiğiniz cevaba göre, müşteriye tavsiyeniz nedir?

# anlamlı bir fark olmadığı için, daha detaylı araştırmalar yapılabilir ve ya maaliyeti
# daha az olan tercih edilebilir