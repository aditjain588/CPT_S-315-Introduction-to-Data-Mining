import pandas as pd
import numpy as np
#from scipy.spatial.distance import cosine

rating_file=pd.read_csv("ratings.csv")
#print(rating_file)
movie_file=pd.read_csv("movies.csv")
#print(movie_file)

df_movie_list=rating_file.pivot(
        index='userId',
        columns='movieId',
        values='rating')


df_movie_list=df_movie_list.fillna(0)
#print(df_movie_list)


similar_mat = pd.DataFrame(index=df_movie_list.columns,columns=df_movie_list.columns)
similar_mat.head()
#print(similar_mat)

#similar_mat=df_movie_list.corr(method = 'pearson', min_periods = 50)
#print(similar_mat)

for i in range(1,len(df_movie_list.columns)) :
    for j in range(i+1,len(df_movie_list.columns)) :
        #print("({},{})".format(i,j))
        dot_product = np.dot(df_movie_list.iloc[:,i], df_movie_list.iloc[:,j])
        n_i = np.linalg.norm(df_movie_list.iloc[:,i])
        n_j = np.linalg.norm(df_movie_list.iloc[:,j])
        x=(dot_product / (n_i * n_j))
        similar_mat[i][j]=x
        #similar_mat.head()
print(similar_mat)
        
with open('output_File.txt','w') as f:

    for k in range(1,len(df_movie_list)-1):
        userratings = df_movie_list.iloc[k]
        print(userratings)
        rec=pd.Series()
    
        for y in range(1,len(userratings)):
            similar = similar_mat[userratings.index[y]].dropna()
            similar = similar.map(lambda x:x * userratings[y])
            print(similar)
            rec = rec.append(similar)
            
            #print("Sorting recommendation")
            rec.sort_values(inplace = True, ascending = False)
            
            a= pd.DataFrame(rec)
            rec_filter = a[~a.index.isin(userratings.index)]
            #print(recommend_filter.head(5))                 
            f.write ("\n User Id "+ str(k)+'\t')
            f.write(' '.join((rec_filter.head(5)).to_string(header=False,
                        index=True,
                        index_names=False).split('\n')))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        