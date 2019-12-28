import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle 
from fuzzywuzzy import fuzz

from gensim.models import FastText
import re
from numpy import savetxt
from gensim.models import FastText

user_listen = pd.read_csv('G:/recommender-api/dataset/Copy of explicit_data - Data preprocessing - songs.csv')
songs = pd.read_csv('G:/recommender-api/dataset/Copy of explicit_data - Songs - All.csv')
songs_with_artist_id = pd.read_csv('G:/recommender-api/dataset/Copy of explicit_data - Songs - All-with artist_id.csv')

class Dynamic_Palylist_Generation:
    def __init__(self):
        self.merged_data = None
        self.model_knn = None
        self.user_id = None 
        self.cooccurence_matrix = None
        self.all_songs = None
        self.popularity_recommendations = None
        
        
    def merge_data(self, user_listen, songs):
        self.merged_data = pd.merge(user_listen, songs.drop_duplicates(['song_id']), on="song_id", how="left")
        self.merged_data['song'] = self.merged_data[['Title', 'Artist']].apply(lambda x: ' - '.join(x), axis=1)
        self.merged_data['listened_song'] = np.ones((441,), dtype=int)
        return self.merged_data
    
    def create(self):
        df_merge = self.merge_data(user_listen, songs)
        # get a count of user_ids for each unique song as recommendation score
        data_grouped = df_merge.groupby(['song_id']).agg({'user_id': 'count'}).reset_index()
        data_grouped.rename(columns = {'user_id': 'score'},inplace=True)

        # Sort the songs based upon recommendation score
        data_sort = data_grouped.sort_values(['score', 'song_id'], ascending = [0,1])

        # Generate a recommendation rank based upon score
        data_sort['Rank'] = data_sort['score'].rank(ascending=0, method='first')

        # Get the top 10 recommendations
        self.popularity_recommendations = data_sort.head(10) 
        return self.popularity_recommendations
    
    def baselineMethod(self): # call this
        df_merge = self.merge_data(user_listen, songs)
        user_recommendations = self.create()
    
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        user_recommendations.reset_index(drop=True, inplace = True)
        popular_song_ids = user_recommendations['song_id']
        return popular_song_ids
    
    # get recommendations based on user favourites
    def user_item_matrix(self, user_listen, songs):
        df_merge = self.merge_data(user_listen, songs)
        user_item_matrix = df_merge.pivot(
            index='song_id',
            columns='user_id',
            values='listened_song'
        ).fillna(0)
        return user_item_matrix
              
    
    def sparse_matrix(self, user_listen, songs):
        df_song_features = self.user_item_matrix(user_listen, songs)
        user_item_mat = csr_matrix(df_song_features.values)
        
        return user_item_mat
    
    def song_idx_mapping(self, user_listen, songs):
        df_song_features = self.user_item_matrix(user_listen, songs)
        song_to_idx = {
            song: i for i, song in 
            enumerate(list(songs.set_index('song_id').loc[df_song_features.index].Title))
        }
        return song_to_idx

    def KNN_model(self, user_listen, songs): # training and saving model # call this and save the model
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        song_user_mat_sparse = self.sparse_matrix(user_listen, songs)
        model_knn.fit(song_user_mat_sparse)
        # save the model
        knnPickle = open('model_knn_for_user_fav', 'wb') 
        pickle.dump(model_knn, knnPickle)   
    
    
    def fuzzy_matching(self, mapper, fav_song, verbose=True):
        match_tuple = []
        # get match
        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), fav_song.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
            return
        if verbose:
            print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
        return match_tuple[0][1]
    
    def DPG_recommendation(self, fav_song): #call this
        n_recommendations = 10
        data = self.sparse_matrix(user_listen, songs)
        mapper = self.song_idx_mapping(user_listen, songs)
        
        # load the model from disk
        loaded_model = pickle.load(open('G:/recommender-api/controller/model_knn_for_user_fav', 'rb'))
        # fit
        loaded_model.fit(data)

        print('You have input song:', fav_song)
        idx = self.fuzzy_matching(mapper, fav_song)

        print('Recommendation system start to make inference')
        print('......\n')
        distances, indices = loaded_model.kneighbors(data[idx], n_neighbors=n_recommendations+1)

        raw_recommends = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        # get reverse mapper
        reverse_mapper = {v: k for k, v in mapper.items()}
        l2 = []
        suggestions =[]
        # print recommendations
        print('Recommendations for {}:'.format(fav_song))
        for i, (idx, dist) in enumerate(raw_recommends):
            # print(reverse_mapper[idx])
            l2.append(songs['song_id'].loc[songs['Title'] == reverse_mapper[idx]]) 
        for i in range(len(l2)):
            for j in l2[i]:
                suggestions.append(j)
            
            #print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))
        print(suggestions)
    
    # get recommendations based on user listening history + suggest new songs
    def get_user_items(self, user_id):
        df_merge = self.merge_data(user_listen, songs)
        user_data = df_merge[df_merge['user_id'] == user_id]
        user_items = list(user_data['song'].unique())
        return user_items
    
    # Get unique users for a given item (song)
    def get_item_users(self, song):
        df_merge = self.merge_data(user_listen, songs)
        item_data = df_merge[df_merge['song'] == song]
        item_users = set(item_data['user_id'].unique())
        return item_users
    
    # Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        df_merge = self.merge_data(user_listen, songs)
        all_items = list(df_merge['song'].unique())
        return all_items

    def get_item_users_by_title(self, Title):
        df_merge = self.merge_data(user_listen, songs)
        item_data = df_merge[df_merge['Title'] == Title]
        item_users_ = set(item_data['user_id'].unique())
        return item_users_     
    
    def get_sentences(self):
        music = songs 
        song_name = music.Title.values
        song_name_clean = [re.sub(r'[^\w]', ' ', str(item))for item in song_name]
        song_name_clean = [re.sub(r" \d+", '', str(item.strip())) for item in song_name_clean]

        sentences = list()
        for item in song_name_clean:
            sentences.append(item.split())
        unique_sentence = np.unique(sentences)
        return unique_sentence
    
    def Fasttext_model(self): # save train content based model
        num_features = 50    # Word vector dimensionality                      
        min_word_count = 1                      
        num_workers = 1      # Number of CPUs
        context = 3          # Context window size; 

        downsampling = 1e-3   # threshold for configuring which 
                              # higher-frequency words are randomly downsampled

        # Initialize and train the model 
        model = FastText(workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling, sg = 1)
        unique_sentence = self.get_sentences()
        model.build_vocab(sentences = unique_sentence)
        model.train(sentences = unique_sentence,  total_examples=len(unique_sentence), epochs=10)

        model.init_sims(replace=True)
        
        model.save('Fasttext.model')
    
    def generate_similars(self, song_name):

        # load the trained model
        model = FastText.load('G:/recommender-api/controller/Fasttext.model')

        # split the song title
        tokens = song_name.split() 
        unique_sentence = self.get_sentences()

        suggestions = []

        # check for most similar items form the model
        suggestions.append(model.wv.most_similar(positive=tokens, topn=10))

        predictions = []
        for l in range(len(suggestions[0])):
            for i in range(len(unique_sentence)):
                for j in range(len(unique_sentence[i])):
                    if unique_sentence[i][j] == suggestions[0][l][0]:
        #                 print(unique_sentence[i])
                        s = ' '
                        word = s.join(unique_sentence[i])
        #                 print(word)
                        predictions.append(word)

        return predictions
    def recommend_new_items(self, user_id, new_song):

        predictions = self.generate_similars(new_song)
        for item in predictions:
            for value in self.get_item_users_by_title(item):
                if value == user_id:
                    return new_song
                else:
                    continue

     # Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):
        df_merge = self.merge_data(user_listen, songs)
        user_songs_users = []
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

            cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        for i in range(0, len(all_songs)):
            # Calculate unique listeners (users) of song (item) i
            songs_i_data = df_merge[df_merge['song'] == all_songs[i]]
            users_i = set(songs_i_data['user_id'].unique())
    #         print(songs_i_data)
    #         print(users_i)

            for j in range(0, len(user_songs)):
                # Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]

                # Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)

                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)

                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))

                else:
                    cooccurence_matrix[j,i] = 0

        return cooccurence_matrix
    
    # Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user_id, all_songs, user_songs, new_song = None):
        df_merge = self.merge_data(user_listen, songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))

        # Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)

        # Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        # index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
        
        # Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user_id,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
            # Handle the case where there are no recommendations
        #print(df)
        l2 = []
        suggestions = []
        for i in df['song']:
            l2.append(df_merge['song_id'].loc[df_merge['song'] == i]) 
        for i in range (len(l2)):
            for j in l2[i]:
                suggestions.append(j)
        suggestions = list(dict.fromkeys(suggestions))
        #print(suggestions)
        if len(suggestions) == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        elif(new_song != None):
            new_song_id = songs['song_id'].loc[songs['Title'] == new_song]
            suggestions.append(new_song_id)
            print(suggestions)
        else:
            print(suggestions)
    

    # Use the item similarity based recommender system model to make recommendations
    def recommend_songs(self, user_id, new_song = None):
        df_merge = self.merge_data(user_listen, songs)
        user_songs = self.get_user_items(user_id)    
        print("No. of unique songs for the user: %d" % len(user_songs))

        all_songs = self.get_all_items_train_data()

        print("no. of unique songs in the training set: %d" % len(all_songs))

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        if (new_song != None):
            new_item = self.recommend_new_items(user_id, new_song)
            df_recommendations = self.generate_top_recommendations(user_id, all_songs, user_songs, new_item)
        else:
            df_recommendations = self.generate_top_recommendations(user_id, all_songs, user_songs)

        return df_recommendations
        

    

