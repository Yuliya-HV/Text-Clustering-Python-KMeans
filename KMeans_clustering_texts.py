# -*- coding: utf-8 -*-
"""
Clustering texts. Extract most relevant words for each cluster.
Data: abstracts from NYTimes newspaper.
"""

import pymysql
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def error_log(id, source, e):
    with open('Error_log.txt', 'a') as file:
        file.write(str(id) + ', ' + str(source) + ', ' + str(e) + ', ' + time.ctime() + '\n')
        file.close()
        
        
conn = pymysql.connect(db="database_name",
                       user="user_name",
                       passwd="password_to_database",
                       host= "host", 
                       port = 3306 #usually this number
                       )
print('Connected to MySQL.')

cursor = conn.cursor()

#select texts for clustering. It could be any texts of your interest.
sql_query_select = "SELECT abstract FROM DT_NYT.NYT_ARTICLE_INFO_3 where length(abstract)>0;"
cursor.execute(sql_query_select)

#load all texts in object 'data'
data = cursor.fetchall()

data_ = []

try:
    for row in data:
        #save all texts in a list for own convenience in further steps 
        data_.append(row[0])
    
    #remove stopwords, vectorize texts
    vectorizer = TfidfVectorizer(stop_words = "english")
    X = vectorizer.fit_transform(data_)
    
    #define number of clusters as 5
    num_k = 5
    
    #build K-Means clustering model
    model = KMeans(n_clusters = num_k, init = 'k-means++', max_iter = 100, n_init = 1)
    model.fit(X)
    
    #extract centroids for each cluster to be able to get a list of top words which are the closest to the centroid
    centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(num_k):
        print("Cluster {}:".format(i))
        for idx in centroids[i, :50]:
            print(terms[idx])
    
except Exception as e:
    print("Error occured: ", e)
finally:
   conn.close()
   print("MySQL connection is closed.")  