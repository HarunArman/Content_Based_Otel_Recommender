import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.birhayalinpesinde.com/wp-content/uploads/2018/02/maldivler_oteller.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


def content_based_recommender(Otel_Adı, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['Otel Adı'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # Otel Adı'ın index'ini yakalama
    movie_index = indices[Otel_Adı]
    # Otel Adı'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:1001].index
    df["Score"] = similarity_scores.sort_values("score", ascending=False)
    return dataframe[['Otel Adı',"Score","Fiyat","il","ilçe","Temalar","Yorum Puan","Images"]].iloc[movie_indices]


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(analyzer='word',stop_words=set(stop))
    dataframe['Yorum'] = dataframe['Yorum'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['Yorum'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim



add_bg_from_url()
df = pd.read_csv("oteller_ve_yorumlar_son.csv",encoding="utf-8")  # DtypeWarning kapamak icin
df["Temalar"] = df["Temalar"].fillna("")
df = df[df['Otel Adı'].notna()]
df = df.reset_index()

yorum_list = []
i_list = []
df_bos = pd.DataFrame()
df_tavsiye_sehir = pd.DataFrame()


for text in df.Yorum:
    text = text.strip()
    text = text.lower()
    yorum_list.append(text)

df = df.assign(Yorum=yorum_list)

file1 = open("turkish stop words.txt",'r')
stop = list(stopwords.words('english'))
stop.extend(file1.readline().split())
file1.close()

koc = {"sehir":["Balıkesir","Muğla"],"tema":["Çocuk Dostu","Açık Havuz"]}
boga = {"sehir":["Antalya", "Bursa", "Erzurum", "Bolu", "Kocaeli"],"tema":["Doğa", "Kayak","Lüks Otel","Butik Otel","Aquapark"]}
ikizler = {"sehir":["İzmir", "Balıkesir", "Mersin"],"tema":["Aquapark", "Denize Sıfır", "Butik Otel"]}
yengec = {"sehir":["İstanbul", "Antalya", "Muğla"],"tema":["Denize sıfır", "Aquapark", "Butik Otel"]}
aslan = {"sehir":["Antalya", "İzmir"],"tema":["Lüks Otel", "Spa"]}
basak = {"sehir":["Nevşehir", "İstanbul", "Kocaeli"],"tema":["Spa", "Doğa"]}
terazi = {"sehir":["Antalya", "Muğla"], "tema":["Lüks Otel","Doğa","Spa"]}
akrep = {"sehir":["Antalya", "İzmir", "Bursa", "Kocaeli", "Erzurum"],"tema":["Denize Sıfır", "Doğa", "Lüks Otel","Kayak"]}
yay = {"sehir":["İzmir","Muğla","Antalya","Balıkesir"],"tema":["Spor", "Aquapark", "Doğa", "Ücretsiz Wifi"]}
oglak = {"sehir":["Antalya", "Muğla","Aydın"], "tema":["Lüks Otel","Doğa","Spa"]}
kova = {"sehir":["Antalya", "İzmir", "Muğla"],"tema":["Lüks otel", "Spa", "Aquapark", "Denize Sıfır"]}
balık = {"sehir":["İzmir", "Muğla", "Bolu", "Balıkesir"],"tema":["Denize sıfır", "Spa", "Lüks Otel"]}


s1 = pd.Series(["Seçiniz"])
appended_series = s1.append(df["Otel Adı"])

st.header('Otel Tavsiye Sistemi')

option_otel = st.selectbox("Daha Önce Konakladığınız Oteli Giriniz",appended_series)
option_burc = st.selectbox("Burcunuzu Giriniz",["Seçiniz","Koç","Boga","İkizler","Yengeç",
                                                "Aslan","Başak","Terazi","Akrep","Yay","Oglak","Kova","Balık"])

sehir_buton = st.checkbox('Burcun Şehir Etkisini Kapat')

burc_dict = {'Koç':koc,'Boga':boga,'İkizler':ikizler,'Yengeç':yengec,'Aslan':aslan,
             'Başak':basak, 'Terazi':terazi, 'Akrep':akrep, 'Yay':yay,'Oglak':oglak,
             'Kova':kova, 'Balık':balık}


if st.button("Tavsiyeleri gör"):
    if option_otel == "Seçiniz":
        st.warning("Lütfen geçerli bir değer seçiniz.")

    elif option_burc == "Seçiniz":
        cosine_sim = calculate_cosine_sim(df)
        df_tavsiye = content_based_recommender(option_otel, cosine_sim, df)

        df_tavsiye = df_tavsiye.sort_values("Score", ascending=False)
        df_tavsiye = df_tavsiye.reset_index()

        images = list(df_tavsiye["Images"])

        for i in range(0,5):
            st.title(str(i+1)+"- "+df_tavsiye["Otel Adı"][i])
            st.image(images[i],width=400)
            st.markdown("""
            <style>
            .big-font {
                font-size:300px !important;
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown('<p><strong><span style="font-size: 22px;">Şehir: </span></strong> <span style="font-size: 22px;">'
                        + str(df_tavsiye["il"][i])+'</span></p>',unsafe_allow_html=True)
            st.markdown('<p><strong><span style="font-size: 22px;">İlçe: </span></strong> <span style="font-size: 22px;">'
                        + str(df_tavsiye["ilçe"][i])+'</span></p>',unsafe_allow_html=True)
            st.markdown(
                '<p><strong><span style="font-size: 22px;">Temalar: </span></strong> <span style="font-size: 22px;">'
                + str(df_tavsiye["Temalar"][i]) + '</span></p>', unsafe_allow_html=True)
            st.markdown(
                '<p><strong><span style="font-size: 22px;">Fiyat: </span></strong> <span style="font-size: 22px;">'
                + str(df_tavsiye["Fiyat"][i]) + ' TL</span></p>', unsafe_allow_html=True)
            st.markdown(
                '<p><strong><span style="font-size: 22px;">Otel Puanı: </span></strong> <span style="font-size: 22px;">'
                + str("{:.2f}".format(df_tavsiye["Yorum Puan"][i])) + '</span></p>', unsafe_allow_html=True)
            st.markdown(
                '<p><strong><span style="font-size: 22px;">Benzerlik Skoru: </span></strong> <span style="font-size: 22px;">'
                + str("{:.2f}".format(df_tavsiye["Score"][i])) + '</span></p>', unsafe_allow_html=True)


    else:

        if sehir_buton:
            cosine_sim = calculate_cosine_sim(df)
            df_tavsiye = content_based_recommender(option_otel, cosine_sim, df)

            a = burc_dict.get(option_burc)

            for i,j in enumerate(df_tavsiye["Temalar"]):
                for k in a["tema"]:
                    if k in j:
                        i_list.append(i)


            df_tavsiye = df_tavsiye[['Otel Adı', "Fiyat", "il","ilçe", "Score","Images","Temalar"]].iloc[i_list]
            df_tavsiye = df_tavsiye.sort_values("Score", ascending=False)
            df_tavsiye = df_tavsiye.reset_index()

            images = list(df_tavsiye["Images"])

            for i in range(0,5):
                st.title(str(i+1)+"- "+df_tavsiye["Otel Adı"][i])
                st.image(images[i],width=400)
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
                    '<p><strong><span style="font-size: 22px;">İlçe: </span></strong> <span style="font-size: 22px;">'
                    + str(df_tavsiye["ilçe"][i]) + '</span></p>', unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">Temalar: </span></strong> <span style="font-size: 22px;">'
                    + str(df_tavsiye["Temalar"][i]) + '</span></p>', unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">Fiyat: </span></strong> <span style="font-size: 22px;">'
                    + str(df_tavsiye["Fiyat"][i]) + ' TL</span></p>', unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">Benzerlik Skoru: </span></strong> <span style="font-size: 22px;">'
                    + str("{:.2f}".format(df_tavsiye["Score"][i])) + '</span></p>', unsafe_allow_html=True)

        else:
            cosine_sim = calculate_cosine_sim(df)
            df_tavsiye = content_based_recommender(option_otel, cosine_sim, df)

            a = burc_dict.get(option_burc)

            for k in a["sehir"]:
                df_tavsiye_sehir = pd.concat([df_tavsiye[df_tavsiye["il"] == k], df_tavsiye_sehir], axis=0)

            df_tavsiye = df_tavsiye_sehir.reset_index()


            for i, j in enumerate(df_tavsiye["Temalar"]):
                for k in a["tema"]:
                    if k in j:
                        i_list.append(i)

            df_tavsiye = df_tavsiye[['Otel Adı', "Fiyat", "il","ilçe", "Score", "Images","Temalar"]].iloc[i_list]
            df_tavsiye = df_tavsiye.sort_values("Score", ascending=False)
            df_tavsiye = df_tavsiye.reset_index()

            images = list(df_tavsiye["Images"])

            for i in range(0,5):
                st.title(str(i + 1) + "- " + df_tavsiye["Otel Adı"][i])
                st.image(images[i], width=400)
                st.markdown("""
                <style>
                .big-font {
                    font-size:300px !important;
                }
                </style>
                """, unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">Şehir: </span></strong> <span style="font-size: 22px;">'
                    + str(df_tavsiye["il"][i]) + '</span></p>', unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">İlçe: </span></strong> <span style="font-size: 22px;">'
                    + str(df_tavsiye["ilçe"][i]) + '</span></p>', unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">Temalar: </span></strong> <span style="font-size: 22px;">'
                    + str(df_tavsiye["Temalar"][i]) + '</span></p>', unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">Fiyat: </span></strong> <span style="font-size: 22px;">'
                    + str(df_tavsiye["Fiyat"][i]) + ' TL</span></p>', unsafe_allow_html=True)
                st.markdown(
                    '<p><strong><span style="font-size: 22px;">Benzerlik Skoru: </span></strong> <span style="font-size: 22px;">'
                    + str("{:.2f}".format(df_tavsiye["Score"][i])) + '</span></p>', unsafe_allow_html=True)

