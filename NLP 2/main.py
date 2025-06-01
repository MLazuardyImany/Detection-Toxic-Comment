import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import re

st.title("Dashboard Analisis Komentar dan Klasifikasi Toxic")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Asli")
    st.write(df.head())

    # Cek kolom
    st.write("Kolom yang tersedia:", df.columns.tolist())

    # Visualisasi kategori
    if 'Label' in df.columns:
        st.subheader("Distribusi Kategori (Label)")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Label', data=df, ax=ax1)
        st.pyplot(fig1)

    # Data Cleansing
    st.subheader("Data Cleansing")

    def clean_text(text):
        text = re.sub(r'http\S+', '', str(text))
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    df['cleaned'] = df['Comment'].apply(clean_text)
    st.write(df[['Comment', 'cleaned']].head())

    # Visualisasi setelah cleansing
    if 'Label' in df.columns:
        st.subheader("Distribusi Kategori Setelah Cleansing")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Label', data=df, ax=ax2)
        st.pyplot(fig2)

    # Wordcloud untuk komentar toxic
    st.subheader("Wordcloud Komentar Toxic")
    toxic_comments = df[df['Label'] == 'toxic']['cleaned'].dropna()
    toxic_text = " ".join(toxic_comments)
    wordcloud = WordCloud(width=800, height=400).generate(toxic_text)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    # Klasifikasi
    st.subheader("Klasifikasi")

    X = df['cleaned']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    # 1. Naive Bayes - Bag of Words
    st.markdown("### Naive Bayes - Bag of Words")
    bow = CountVectorizer()
    X_train_bow = bow.fit_transform(X_train)
    X_test_bow = bow.transform(X_test)
    nb_bow = MultinomialNB()
    nb_bow.fit(X_train_bow, y_train)
    y_pred_bow = nb_bow.predict(X_test_bow)
    acc_bow = accuracy_score(y_test, y_pred_bow)
    st.write("Akurasi:", acc_bow)
    results['NB_BOW'] = acc_bow

    # 2. Naive Bayes - TF-IDF
    st.markdown("### Naive Bayes - TF-IDF")
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
    acc_tfidf = accuracy_score(y_test, y_pred_tfidf)
    st.write("Akurasi:", acc_tfidf)
    results['NB_TFIDF'] = acc_tfidf

    # 3. Decision Tree
    st.markdown("### Decision Tree")
    dt = DecisionTreeClassifier()
    dt.fit(X_train_tfidf, y_train)  # Pakai TF-IDF
    y_pred_dt = dt.predict(X_test_tfidf)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    st.write("Akurasi:", acc_dt)
    results['DecisionTree'] = acc_dt

    # Plot perbandingan model dengan tipe
    st.subheader("Perbandingan Akurasi Model")
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Akurasi': list(results.values()),
        'Tipe': ['Klasifikasi'] * len(results)
    })

    fig_comp, ax_comp = plt.subplots()
    sns.barplot(data=results_df, x='Model', y='Akurasi', hue='Tipe', ax=ax_comp)
    ax_comp.set_ylim(0, 1)
    ax_comp.set_ylabel("Akurasi")
    ax_comp.set_title("Perbandingan Akurasi Model Berdasarkan Tipe")
    st.pyplot(fig_comp)

    st.success("Proses selesai!")
