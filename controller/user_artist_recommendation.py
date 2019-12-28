import pandas as pd
import numpy as numpy

artists = pd.read_csv('G:/recommender-api/dataset/Copy of explicit_data - Artists - All.csv')
user_listening = pd.read_csv('G:/recommender-api/dataset/copy of explicit_data - DATA preprocessing - artist.csv')
user_songs = pd.read_csv('G:/recommender-api/dataset/Copy of explicit_data - Data preprocessing - songs.csv')
songs = pd.read_csv('G:/recommender-api/dataset/Copy of explicit_data - Songs - All-with artist_id.csv')


class recommender:
    def __init__(self):
        self.user_songs_data = None
        self.user_artists_data = None
        self.user_id = None
        self.item_id = None
        self.artist_id = None
        
    def create_user_song_pref(self, user_songs, songs):
        user_song_pref = pd.merge(user_songs, songs.drop_duplicates(['song_id']), on="song_id", how="left")
        user_song_pref = user_song_pref.drop(['Album','Genre'], axis=1)
        return user_song_pref

    def create_user_rtist_pref(self, user_listening, artists):
        user_artist_pref = pd.merge(user_listening, artists.drop_duplicates(['artist_id']), on="artist_id", how="left")
        user_artist_pref = user_artist_pref.drop(['preferred_artists','musical_aspect','Unnamed: 9'], axis=1)
        return user_artist_pref
    
    def create_score(self, user_songs_data):
        train_data_grouped = user_songs_data.groupby(['song_id','Artist_id','Title']).agg({'user_id': 'count'}).reset_index()
        train_data_grouped.rename(columns = {'user_id': 'score'},inplace=True)
        # sort the values to get an overview of the popular songs
        train_data_sort = train_data_grouped.sort_values(['score', 'song_id'], ascending = [0,1])
        train_data_sort = pd.DataFrame(train_data_sort)
        train_data_sort.reset_index(drop=True, inplace = True)
        return train_data_sort

    def recommend_songs(self, user_id):
        user_artists_data = self.create_user_rtist_pref(user_listening, artists)
        user_songs_data = self.create_user_song_pref(user_songs, songs)
        songs_ = []
        scores = []
        l2 = []
        l3 = []
        user_pref = user_artists_data.loc[user_artists_data['user_id'] == user_id]
        createScore = self.create_score(user_songs_data)
        for i in user_pref['artist_id']:
            songs_.append(createScore['Title'].loc[createScore['Artist_id'] == i])
            scores.append(createScore['score'].loc[createScore['Artist_id'] == i])
        for i in range(len(songs_)):
            for j in songs_[i]:
                l2.append(j)
        for i in range(len(scores)):
            for j in scores[i]:
                l3.append(j)
        list_of_tuples = list(zip(l2, l3)) 
        df = pd.DataFrame(list_of_tuples, columns = ['Songs', 'Score'])  
        sort_by_life = df.sort_values('Score', ascending=False)
        sort_by_life.reset_index(drop=True, inplace = True)
        sugesions = []
        idx = []
        for i in sort_by_life['Songs']:
            idx.append(songs['song_id'].loc[songs['Title'] == i])
        for i in range(len(idx)):
            for j in idx[i]:
                sugesions.append(j)
        return sugesions

        
