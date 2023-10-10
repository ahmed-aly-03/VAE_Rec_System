#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


columnNames = ['User_id','MovieID','Rating']

MovieLens = pd.read_csv("/Users/ahmedaly/Desktop/Final_Model/ml-100k/u.data", sep = '\t', names = columnNames, 
        engine = 'python',usecols = [0, 1, 2]) 

MovieLens = MovieLens.sort_values(by = 'MovieID')


# In[3]:


columnNames = ['MovieID','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama'
              ,'Fantasy','FilmNoir','Horror','Musical','Mystery','Romance','SciFi','Thriller','War','Western']

useCols = [0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

genreInformation = pd.read_csv('/Users/ahmedaly/Desktop/Final_Model/ml-100k/u.item', sep = '|',
        names = columnNames, encoding = 'latin-1', usecols = useCols)


# In[4]:


MovieLens = pd.merge(genreInformation,MovieLens, on = 'MovieID')

columnOrder = ['User_id', 'MovieID','Rating','unknown','Action','Adventure','Animation','Childrens',
               'Comedy','Crime','Documentary','Drama','Fantasy','FilmNoir','Horror','Musical','Mystery'
               ,'Romance','SciFi','Thriller','War','Western']

MovieLens = MovieLens[columnOrder]


# In[5]:


class InteractionMatrix:
    def __init__ (self,domainA,domainB):
        
        columns = ['User_id', 'MovieID', 'Rating', domainA, domainB]
        
        Movie_Lens_selected_columns = MovieLens[columns] 
        
        self.RatingMatrix = Movie_Lens_selected_columns[(Movie_Lens_selected_columns[domainA] == 1) | 
                (Movie_Lens_selected_columns[domainB] == 1)]
        
        self.RatingMatrix = self.RatingMatrix.groupby('User_id').filter(
            lambda x: x[domainA].sum() > 0 and x[domainB].sum() > 0)
        
        self.domainARatingMatrix = self.RatingMatrix[self.RatingMatrix[domainA] == 1 ]
        
        self.domainBRatingMatrix = self.RatingMatrix[self.RatingMatrix[domainB] == 1 ]

        self.domainARatingMatrix = self.domainARatingMatrix.pivot_table(index = 'User_id', columns = 'MovieID',
                                                                        values = 'Rating').fillna(0)

        self.domainBRatingMatrix = self.domainBRatingMatrix.pivot_table(index = 'User_id', columns = 'MovieID',
                                                                        values = 'Rating').fillna(0)
        
        self.RatingMatrix = self.RatingMatrix.pivot_table(index = 'User_id', columns = 'MovieID', 
                                                                        values = 'Rating' ).fillna(0)

