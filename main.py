#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Since our dataset is huge, we will use sample of size 40000

# In[38]:


users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv").sample(n=40000, random_state=42)
books = pd.read_csv("Books.csv")


# In[39]:


pd.set_option('display.max_columns',10)


# In[40]:


ratings


# In[42]:


ratings.describe()


# 

# In[43]:


plt.hist(ratings["Book-Rating"])


# In[44]:


print(ratings["ISBN"].nunique())


# Now we would like to create matrix with ISBN and it's corresponding ratings, however pivot_table is still returning some kind of error. Probably some memory issue. First we will remove all the 0 ratings. There are some ISBN values in ratings, that are not in books dataset, we will also remove them.

# In[132]:


ratings_filtered = ratings[ratings["Book-Rating"] != 0]
mask = ratings_filtered['ISBN'].isin(books['ISBN'])
ratings_filtered = ratings_filtered[mask]


# In[70]:


ratings_filtered.describe()


# In[47]:


ratings_filtered


# In[135]:


ratings_filtered['ISBN'].nunique()


# In[143]:


selected_isbn = ratings_filtered['ISBN'].unique()
selected_books = books[books['ISBN'].isin(selected_isbn)]

#print(selected_books['Book-Title'])
# In[144]:


selected_books


# Let's create the item-user matrix.

# In[72]:


pivot_table = ratings_filtered.pivot_table(index="ISBN",columns="User-ID",values="Book-Rating").fillna(0)


# In[73]:


len(pivot_table.index)


# In[74]:


user_matrix = pivot_table.to_numpy()
user_matrix


# Let's check for matrix sparsity. Collaborative filtering do NOT work well if data is too sparse.

# In[148]:


sparsity = np.count_nonzero(user_matrix)/(np.shape(user_matrix)[0]*np.shape(user_matrix)[1])
sparsity


# The dataset is really sparse, let's continue for now and later we might remove books that have been rated only few times and see if that improves our resulting model.
# 

# In[76]:


from sklearn.decomposition import TruncatedSVD


# In[77]:


svd = TruncatedSVD(15)

result = svd.fit_transform(user_matrix)


# In[78]:


result


# In[79]:


user_matrix_reconstructed = np.dot(result,svd.components_)


# In[57]:


user_matrix_reconstructed


# In[80]:


df_user_matrix = pd.DataFrame(data=user_matrix_reconstructed,index = pivot_table.index)
df_user_matrix


# In[88]:


from sklearn.metrics.pairwise import cosine_similarity


#Funcion that computes cosine similarity of given book with each other in dataset. Returns list where each element is cosine similarity with given book and it's ISBN.
def similar(book):
    isbn = books.loc[books['Book-Title'] == book]['ISBN'].iloc[0]
    similarities = []
    vector = df_user_matrix.loc[isbn].values.reshape(1,-1)
    for index in df_user_matrix.index:
        similarities.append([cosine_similarity(vector,df_user_matrix.loc[index].values.reshape(1,-1))[0][0],index])
    
    return similarities
        


# In[121]:


#This function takes similarity measure function, book and returns list of top5 similar books with it's ISBN.
def recommend(fun,book):
    arr = np.array(fun(book))
    indices_sorted = np.argsort(np.float_(arr[:,0]))
    top5 = arr[indices_sorted][-5:]
    predicted_books = [books[books['ISBN'] == isbn]['Book-Title'] for isbn in top5[:,1]]
    return predicted_books
    
    


# In[119]:





# In[120]:





# In[145]:


another_pred = recommend(similar,'Timeline')


# In[146]:


print((another_pred[0]))

# Now we need to select scoring metric and perform some kind of hyperparameter selection. Hyper parameters might be different matrix factorization algorithms, parameters of these algorithms, similarity functions and any kind of data preparation. We might scale ratings to eliminate users rating bias, we might remove users/books that rated/have been rated only few times and so on...

# 
