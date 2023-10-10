#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


columns_to_use = [0,1,2]

column_names = ['User_id', 'MovieID', 'Rating']

MovieLens1M = pd.read_csv('/Users/ahmedaly/Desktop/Final_Model/ml-1m/ratings.dat', sep = '::', names = column_names,
                          engine = 'python', usecols = columns_to_use)

MovieLens1M = MovieLens1M.sort_values(by = 'MovieID')


# In[3]:


column_names_2 = ['MovieID', 'Genres']#checked for null columns and duplicates there were none

movie_genres = pd.read_csv('/Users/ahmedaly/Desktop/Final_Model/ml-1m/movies.dat', sep = '::', names = column_names_2,
                            encoding = 'latin-1', engine = 'python', usecols = [0,2])


# In[4]:


binary_movie_genres = movie_genres['Genres'].str.get_dummies('|')

movie_genres = pd.concat([movie_genres, binary_movie_genres], axis = 1)

movie_genres.drop(['Genres'], axis = 1, inplace = True)


# In[5]:


MovieLens1M = pd.merge(MovieLens1M, movie_genres, on = 'MovieID')


# In[6]:


class RatingMatrix_ML1M:
    def __init__ (self,domainA,domainB):
        
        columns = ['User_id', 'MovieID', 'Rating', domainA, domainB]
        
        Movie_Lens_selected_columns = MovieLens1M[columns] 
        
        self.ratingMatrix = Movie_Lens_selected_columns[(Movie_Lens_selected_columns[domainA] == 1) | 
                (Movie_Lens_selected_columns[domainB] == 1)]
        
        self.ratingMatrix = self.ratingMatrix.groupby('User_id').filter(
            lambda x: x[domainA].sum() > 0 and x[domainB].sum() > 0)
        
        self.domainARatingMatrix = self.ratingMatrix[self.ratingMatrix[domainA] == 1 ]
        
        self.domainBRatingMatrix = self.ratingMatrix[self.ratingMatrix[domainB] == 1 ]

        self.domainARatingMatrix = self.domainARatingMatrix.pivot_table(index = 'User_id', columns = 'MovieID',
                                                                        values = 'Rating').fillna(0)

        self.domainBRatingMatrix = self.domainBRatingMatrix.pivot_table(index = 'User_id', columns = 'MovieID',
                                                                        values = 'Rating').fillna(0)
        
        self.ratingMatrix = self.ratingMatrix.pivot_table(index = 'User_id', columns = 'MovieID', 
                                                                        values = 'Rating' ).fillna(0)

