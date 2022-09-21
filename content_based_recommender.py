import pandas as pd
import numpy as np
import streamlit as st
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("temalar_yorumlar.csv",encoding="utf-8")  # DtypeWarning kapamak icin
df["Temalar"] = df["Temalar"].fillna("")

koc = {"sehir":["Balıkesir","Muğla"],"tema":["Çocuk Dostu","Açık Havuz"]}
i_list = []
df_bos = pd.DataFrame()
df_tavsiye_sehir = pd.DataFrame()


def content_based_recommender(Otel_Adı, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['Otel Adı'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # Otel Adı'ın index'ini yakalama
    movie_index = indices[Otel_Adı]
    # Otel Adı'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:201].index
    df["Score"] = similarity_scores.sort_values("score", ascending=False)
    return dataframe[['Otel Adı',"Score","Fiyat","il","Temalar","Images"]].iloc[movie_indices]



def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['Yorum'] = dataframe['Yorum'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['Yorum'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

type(df["Otel Adı"])
s1 = pd.Series(["Seçiniz"])
appended_series = s1.append(df["Otel Adı"])
st.header('Otel Tavsiye Sistemi')
option_otel = st.selectbox("Daha Önce Konakladığınız Oteli Giriniz",appended_series)
option_burc = st.selectbox("Burcunuzu Giriniz",["Seçiniz","Koç","Boğa","Yengeç","Terazi",])

if st.button("Tavsiyeleri gör"):
    if option_otel == "Seçiniz" or option_burc == "Seçiniz":
        st.warning("Lütfen geçerli bir değer seçiniz.")
    else:
        cosine_sim = calculate_cosine_sim(df)
        df_tavsiye = content_based_recommender(option_otel,cosine_sim,df)

        if option_burc == "Koç":
            for k in koc["sehir"]:
                df_tavsiye_sehir = pd.concat([df_tavsiye[df_tavsiye["il"]==k],df_tavsiye_sehir],axis=0)


        df_tavsiye = df_tavsiye_sehir.reset_index()


        for i,j in enumerate(df_tavsiye["Temalar"]):
            for k in koc["tema"]:
                if k in j:
                    i_list.append(i)


        df_tavsiye = df_tavsiye[['Otel Adı', "Fiyat", "il", "Score","Images"]].iloc[i_list]
        df_tavsiye = df_tavsiye.reset_index()

        images = list(df_tavsiye["Images"])

        for i in range(len(df_tavsiye.index)):
            st.title(str(i+1)+"- "+df_tavsiye["Otel Adı"][i])
            st.image(images[i],width=800)
            st.markdown("""
            <style>
            .big-font {
                font-size:300px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown('<p><strong><span style="font-size: 22px;">Şehir: </span></strong> <span style="font-size: 22px;">'
                        + str(df_tavsiye["il"][i])+'</span></p>',unsafe_allow_html=True)
            st.markdown(
                '<p><strong><span style="font-size: 22px;">Fiyat: </span></strong> <span style="font-size: 22px;">'
                + str(df_tavsiye["Fiyat"][i]) + ' TL</span></p>', unsafe_allow_html=True)
            st.markdown(
                '<p><strong><span style="font-size: 22px;">Benzerlik Skoru: </span></strong> <span style="font-size: 22px;">'
                + str("{:.2f}".format(df_tavsiye["Score"][i])) + '</span></p>', unsafe_allow_html=True)







            #df_tavsiye[['Otel Adı', "Fiyat", "il", "Score"]].iloc[i_list]
            #st.image(
               # "https://cdn.tatilsepeti.com/Files/TesisResim/07742/tsr07742636958442028934481.jpg",
              #  width=800,  # Manually Adjust the width of the image as per requirement
            #)
