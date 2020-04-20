#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 13:03:52 2019

@author: Gyorgy Attila Ruzicska

Project for the Text Analysis with Python course

In this notebook, I will analyse the similarities of the 100 most popular books
from the Project Gutenberg (gutenberg.org).
In particular, I will make some analysis from the viewpoint of my favourite novel:
'Les Miserables'.
"""

# Import libraries
import glob, re, os
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim import similarities
from gensim.models import TfidfModel
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.hierarchy import dendrogram
from sklearn.feature_extraction.text import TfidfVectorizer


# The books files are contained in this folder
folder = "/Users/apple/Desktop/CEU/TextAnalysis/Project/Datasets/"

# List all the .txt files and sort them alphabetically
files = glob.glob(folder+"*.txt")
files.sort()

"""
I will do the following preprocessing steps:
    Removing non-alpha-numeric characters
    Lower casing characters
    Tokenization
    Removing stopwords
    Porter Stemming
"""

# Initialize the object that will contain the texts and titles
txts = []
tokenized_txts=[]
titles = []

#Define a set of stopwords provided by NLTK library
stoplist = set(stopwords.words("english")) 

for n in files:
    #Open the files
    f = open(n, encoding='utf-8-sig')
    #Remove non-alphabetic characters
    data = re.sub('[^a-zA-Z]+', ' ', f.read())
    txts.append(data)
    #Lowercase texts
    data = data.lower()
    #Tokenize the text
    data = data.split()
    #Remove stopwords
    data = [word for word in data if word not in stoplist]
    # Create an instance of a PorterStemmer object
    porter = PorterStemmer()
    # For each token of each text, we generated its stem 
    texts_stem = [porter.stem(token) for token in data]
    #Add the tokens to the initialized object
    tokenized_txts.append(texts_stem)
    #Add the title to the titles object
    titles.append(os.path.basename(n).replace(".txt", ""))

"""
Next, I will create a bag of words, to be used later when computing text similarity
"""

# Create a dictionary from the stemmed tokens
dictionary = corpora.Dictionary(tokenized_txts)

# Create a bag-of-words model for each book, using the previously generated dictionary
bows = [dictionary.doc2bow(text) for text in tokenized_txts]

"""
The following code create the TF-IDF model and displays it in a nice dataframe format
This will later be used in the specific analysis of 'Les Misérables'
"""

#Create a word list as inputs to the tf-idf model
word_list=[]
#Join the list of tokens to string
for element in tokenized_txts:
    words=' '.join(element)
    word_list.append(words)
    
#Initialize the function and fit-transform the word list
vectorizer = TfidfVectorizer()
doc_vector = vectorizer.fit_transform(word_list)

#Create a dataframe from the tf-idf model
df_tfidf = pd.DataFrame(doc_vector.toarray().transpose(), index=vectorizer.get_feature_names(),\
                        columns = titles)

"""
Compute the distance between texts using cosine similiarity
"""

# Generate the tf-idf model from the bag of words model
model = TfidfModel(bows)

# Compute the similarity matrix (pairwise distance between all texts)
sims = similarities.MatrixSimilarity(model[bows])

# Transform the resulting list into a dataframe
sim_df = pd.DataFrame(list(sims))

# Add the titles of the books as columns and index of the dataframe
sim_df.columns=[title for title in titles]
sim_df.index = [title for title in titles]

"""
With the above code, I am able to produce a program which
takes a book name as input and outputs the book which has the highest text
similarity to it.
"""

#Generate an input from users
book_title=input("Input your preferred book: ")

#Try to obtain from the similarity matrix the most similar book, otherwise print an error message
try:
    closest=pd.DataFrame(sim_df[book_title].nlargest(n=2))
    closest.iloc[1,0]
    closest.index[1]
    print("The closest book from my dataset is: " +str(closest.index[1])+" with similarity index: " +str(closest.iloc[1,0]))
except:
    print("The name of the book has been mistyped or it is not contained in my dataset")



"""
In the next sections, I will apply some general analyses about the dataset. This
includes displaying text similarity using dendrograms and applying K-means and
MiniBatch K-Means Clustering.

I use dendograms to display all the information about book similarities at once.
"""

# Compute the clusters from the similarity matrix, using the complete variance minimization algorithm
Z = hierarchy.linkage(sim_df,'complete')

# Plot the dendrogram, using title as label column
dendrogram_ = dendrogram(Z, orientation='top', labels=sim_df.index, leaf_rotation=90, leaf_font_size=16)

# Adjust the plot
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

# Save the plotted dendrogram
plt.savefig('dendrogram.png')


"""
I use KMeans and MiniBatch KMeans Clustering to further analyse how books are related to
one another
"""
#First determine the optimal number of clusters using the elbow method and
# the within group sum of squares

#Initialize a list which will include the within group sum of squares for
# cluster numbers 1-40
wgss = []
#Compute the wgss for each cluster number and add to the list
for i in range(1, 40):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter=100, random_state = 42)
    kmeans.fit(doc_vector)
    wgss.append(kmeans.inertia_)
    
#Plot the results stored in the list, from the picture we cannot conclude the
# optimal number of clusters, therefore, I will take it 15
plt.plot(range(1, 40), wgss)
plt.title('Within group sum of squares')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')

# Save the plotted figure
plt.savefig('elbow.png')

# Create a KMeans object with 15 clusters
km = KMeans(n_clusters=15, init = 'k-means++', max_iter=100, random_state = 17)

# Fit the k-means object with tfidf_matrix
km.fit(doc_vector)

km_clusters = km.labels_.tolist()

#Create dataframe from the titles and clusters
df_km_cluster=pd.DataFrame(list(zip(titles,km_clusters)), columns=['Titles','Clusters'])
        

# MiniBatchKMeans runs k-means in batch mode suitable for a very large corpus
kmini = MiniBatchKMeans(n_clusters=15, init='k-means++', n_init=1,
                        init_size=1000,
                       batch_size=1000
                       )

kmini.fit(doc_vector)

kmini_clusters = kmini.labels_.tolist()


#Create dataframe from the titles and clusters
df_kmini_cluster=pd.DataFrame(list(zip(titles,kmini_clusters)), columns=['Titles','Clusters'])



"""
In the last part of the general analysis, I test the clustering model to another
book that is not part of the corpus and see which cluster it is predicted to belong to.
"""

#Do the same pre-processing steps
odyssey = open("/Users/apple/Desktop/CEU/TextAnalysis/Project/Odyssey.txt", encoding='utf-8-sig')
data = re.sub('[^a-zA-Z]+', ' ', odyssey.read())
data = data.lower()
data = data.split()
data = [word for word in data if word not in stoplist]
ody_stem = [porter.stem(token) for token in data]

words_ody=[]
words=' '.join(ody_stem)
words_ody.append(words)

#Apply the tf-idf vectorizer
ody = vectorizer.transform(words_ody)

#Predict cluster using K-Means
prediction1 = km.predict(ody)
print(prediction1)

#Predict cluster using Mini Batch K-Means
prediction2 = kmini.predict(ody)
print(prediction2)


"""
Next, I will perform some analysis focusing on the book: 'Les Miserables'

First, I check which words are the most common in 'Les Misérables' as weighted by TF-IDF
"""

#Get the relevant column and sort by descending value
freq = df_tfidf.loc[:,'Les Misérables'].sort_values(ascending=False, inplace=False)
#Keep the 10 most frequent
freq_10=freq[:10]
words = freq_10.index

#Plot the most frequent words
xvalues = np.arange(len(freq_10.index))
ax = plt.axes(frameon=True)
ax.set_xticks(xvalues)
ax.set_xticklabels(words, rotation='vertical', fontsize=9)
ax.set_title('Word Frequency Chart')
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
plt.bar(xvalues,freq_10, 1.0 )
plt.savefig('miserables_words.png')

"""
Second, I check  which books are the closest to 'Les Misérables'
"""

# Select the column corresponding to 'Les Misérables'
v = sim_df["Les Misérables"]

# Sort by ascending value
v_sorted = v.sort_values()

#Drop Les Misérables
v_sorted=v_sorted.drop('Les Misérables')

#Drop the least similar books
v_sorted=v_sorted.iloc[79:]

# Plot this data
v_sorted.plot.barh(x='lab', y='val', rot=0).plot()
plt.xlabel("Similarity")
plt.ylabel("Books")
plt.title("Books similar to Les Misérables")
plt.savefig('miserables_similar.png')







































