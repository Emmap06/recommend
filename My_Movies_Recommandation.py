
#ZEZE OTHNIEL AIME
# APPLICATION DE RECOMMANDATION DE FILM

# Importation des librairies
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.corpus import stopwords
import pickle
import streamlit as st
import webbrowser
from streamlit import session_state as session
from streamlit import components
from flask import Flask, request, jsonify
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

#Importation des bases de données
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Copie du dataframedans un nouveau dataframe
moviesWithGenres_df = movies_df.copy()
    
# Pour chaque ligne du dataframe, parcourir la liste de genres et placer 1 à la colonne correspondante du nouveau dataframe
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1
            
# Remplacer les valeurs NaNpar des 0 pour indiquer qu'un film n'est pas de ce genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()
    
# Supprimer une colonne ou une ligne spécifique d'un dataframe
ratings_df = ratings_df.drop('timestamp', axis= 1)

#Definir la fonction de recommandation (basé sur le contenu)
def Movies_Recommandations(userInput, ratings_df, moviesWithGenres_df, n_recommandations=10):
    
    #Extraction des films notés par l'utilisateur et leurs notes
    userInput = [
        {"movieId": title, "rating": rating}
        for title, rating in zip(ratings_df["movieId"], ratings_df["rating"])
    ]
          
    #Les notes attribuées par l'utilisateur
    inputMovies= pd.DataFrame(userInput)

    #Filtrer les films sur la base des titres
    inputId= movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

    #Fusionner de façon implicite sur la base des titre, pour avoir notre dataframe
    inputMovies= pd.merge(inputId, inputMovies)

    # Supprimer les colonnes dont nous n'avons pas besoin dans notre dataframepour libérer de la mémoire
    inputMovies= inputMovies.drop('genres').drop('year')

    # Filtrer les films
    userMovies= moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

    # Réinitialisation de l'index
    userMovies= userMovies.reset_index(drop=True)

    # Supprimer les colonnes non nécessaires
    userGenreTable= userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

    # Produit matriciel pour obtenir les poids
    userProfile= userGenreTable.transpose().dot(inputMovies['rating'])

    # Récupérons les genres de chaque film de notre dataframed'origine
    genreTable= moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

    # Et supprimons less colonnes non nécessaires
    genreTable= genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

    # Multiplier les genres par les poids et calculer la moyenne pondérée
    recommendationTable_df= ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

    # Ordonner les recommandations par ordre décroissant
    recommendationTable_df= recommendationTable_df.sort_values(ascending=False)

    # Le résultatfinal // On va chercher des films
    recommanded_movies = movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]
    return recommanded_movies[:n_recommandations]


# Définir La fonction de l'Application
def MoviesRecommandationsApp():
    #Attribuer un nom à la page
    st.set_page_config(page_title='Theater Home Movies', page_icon='(^_^)')
    
    #Définir des polices d'écritures pour les titre et sous-titre
    st.markdown(
        """
        <style>
        h1 {
            Font_family: "Arial", sans-serif;
            color: #00008B;
        }
        
        h2 {
            Font_family: "Courier New", monospace;
            color: #808080;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    #Bouton de changement de thème (à titre décoratif)
    with st.sidebar:
        if st.button("Mode Nuit / Jour"):
            if st.get_option('theme') == 'light':
                st.get_theme('dark')
            else:
               st.set_theme('light')
    
    #Titre et sous-titre de l'application
    st.sidebar.markdown('# Theater Home Movies')
    st.sidebar.subheader('Welcome on our App')
    
    #importer des images
    st.sidebar.image('https://www.nextplz.fr/wp-content/uploads/nextplz/2022/01/fast-and-furious-10-1200x675.jpg', use_column_width=True)
    st.sidebar.image('https://lumiere-a.akamaihd.net/v1/images/movie_poster_zootopia_866a1bf2.jpeg?region=0%2C0%2C300%2C450', use_column_width=True)
    st.sidebar.image('https://img.phonandroid.com/2021/04/avengers-infinity-war.jpg', use_column_width=True)
    
    #Message pour utilisateur
    st.subheader('Veuillez attribuer des notes aux films: ')
    userInput = []
    for i in range(5):
        title = st.text_input(f"Titre du film {i+1}")
        rating = st.select_slider(f"Note du film {i+1}", options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        if title and rating:
            userInput.append({"titre": title, "Note": rating})

    #Appliquer la fonction de recommandation
    if st.button("Recommandations"):
        if userInput:
            recommanded_movies = Movies_Recommandations(userInput, ratings_df, moviesWithGenres_df, n_recommandations=10)
            st.subheader("Les Films recommandés")
            st.dataframe(recommanded_movies[['title', 'genres']])
        else:
            st.warning("Entrez invalide. Veuillez Vérifier vos entrées précédentes")
    
    #Adresse e-mail
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Pour plus d'info, Envoyer-nous un e-mail"):
            mailto_link = "mailto:aimezez02@gmail.com"
            webbrowser.open_new(mailto_link)


if __name__ == '__main__':
    MoviesRecommandationsApp()

