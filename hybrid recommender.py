import pandas as pd
pd.set_option('display.max_columns', 20)

####### 1.görev = VERİNİN HAZIRLANMASI #####

movie = pd.read_csv("movie.csv")
movie.head()
movie.shape

ratings = pd.read_csv("rating.csv")
ratings.shape

ratings.head(15)
df = movie.merge(ratings, how="left", on="movieId")
df.tail(15)

### 6. Görevde yer alan İTEM-BASED öneriyi burada gerçekleştirdim ###

# user-movie matrisinin oluşturulması
# öneri yapmadan önce bazı filtrelemeler yapmak uygun olacaktır, mesela çok düşük puanlı olan ürünlerin elenmesi

df["title"].value_counts().head()  # her bir filme yapılan değerlendirme
df["title"].value_counts().tail() ## en az yorum alanlar
## belli bir seviye üzeri (3000 ve üzeri) yorum yapılanları seçtim
choose_movies = pd.DataFrame(df["title"].value_counts())
great_movies = choose_movies[choose_movies["title"] >= 3000].index
great_movies = df[df["title"].isin(great_movies)]
great_movies.shape
great_movies["title"].nunique()  ## kalan eşşiz film sayısı
user_movie_df = great_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
## index e sürunları aldım, colonlarda isimler, values da rating olsun
user_movie_df.head(10)
movie_name = "Ocean's Twelve (2004)"   ##input("enter the Moive Name")
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(5)



####### 2.görev = Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi #####

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user_df = user_movie_df[user_movie_df.index == random_user]
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched  ## İzlediği filmler
user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List (1993)"]
len(movies_watched)  ## kaç tane film izlemiş #19


movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]
user_movie_count[user_movie_count["movie_count"] > 10].sort_values("movie_count", ascending=False)
user_movie_count[user_movie_count["movie_count"] == 20].count() ## hedef userın izlediği total film sayısı 20, onunla birebir tüm filmleri izleyen kişi sayısı
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 10]["userId"]
users_same_movies.head()
users_same_movies.count() ## 2414

####### 3.görev = Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişim########
####### 4.görev = Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleme ########

## **Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi**
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      random_user_df[movies_watched]])
final_df.head()
final_df.shape

final_df.T.corr()
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()


# Hedef user ile yüzde 65 ve üzeri korelasyona sahip kullanıcılar:
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)


top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
## user ile ortak zevke sahip kullanıcılar bulduktan sonra, ortak zevkli kullanıların başka hangi filmler begendiklerinin bulunması
rating = pd.read_csv('4.hafta/datasets/ratings.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

####### 5.görev = Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutma ########


top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df[["movieId"]].nunique()
recommendation_df[["movieId"]].head(5)

####### 6.görev =
# ▪ 5 öneri user-based
# ▪ 5 öneri item-based olacak şekilde 10 öneri yapınız########

## skoru 4 ten büyük olanların tavsiye edilmesi
recommendation_df[recommendation_df["weighted_rating"] > 4]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4]

## öneri film isimlerinin gelmesi
movie = pd.read_csv('4.hafta/datasets/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]]).head()
movies_to_be_recommend.head()

## USER BASED =
# Seven (a.k.a. Se7en) (1995)
# Usual Suspects, The (1995)
# Pulp Fiction (1994)
# Crow, The (1994)
# Blade Runner (1982)

## iTEM BASED =
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(5)

#Ocean's Twelve (2004)                              1.000000
#Ocean's Eleven (2001)                              0.550722
#Pirates of the Caribbean: At World's End (2007)    0.459886
#Mr. & Mrs. Smith (2005)                            0.456196
#Bourne Supremacy, The (2004)



