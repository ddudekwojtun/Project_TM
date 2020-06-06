import pandas as pd
import numpy as np
import os
import urllib.request
import glob
import matplotlib.style
import matplotlib.pyplot as plt
from io import open
import codecs
from IPython.display import display

# Just making the plots look better
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 12
plt.xlabel("Word", labelpad=14)
plt.ylabel("Number of Words", labelpad=14)
plt.title("Text analysis in Pandas with TF-IDF", y=1.02)

if not os.path.exists('data'):
     os.mkdir('data')
     urllib.request.urlretrieve('https://www.gutenberg.org/files/2591/2591-0.txt', 'data/grimms_fairy_tales.txt')
     urllib.request.urlretrieve('http://www.gutenberg.org/cache/epub/345/pg345.txt', 'data/dracula.txt')
     urllib.request.urlretrieve('http://www.gutenberg.org/files/2701/2701-0.txt', 'data/moby_dick.txt')
     urllib.request.urlretrieve('http://www.gutenberg.org/files/74/74-0.txt', 'data/tom_sawyer.txt')
     urllib.request.urlretrieve('http://www.gutenberg.org/files/2600/2600-0.txt', 'data/war_and_peace.txt');

books = glob.glob('data/*.txt')
d = list()
for book_file in books:
    with codecs.open(book_file, "r", encoding='utf-8', errors='ignore') as f:
        book = os.path.basename(book_file.split('.')[0])
        d.append(pd.DataFrame({'book': book, 'lines': f.readlines()}))
doc = pd.concat(d)
doc.head()
doc['book'].value_counts().plot.bar()

doc['words'] = doc.lines.str.strip().str.split('[\W_]+')
doc.head()

rows = list()
for row in doc[['book', 'words']].iterrows():
    r = row[1]
    for word in r.words:
        rows.append((r.book, word))

words = pd.DataFrame(rows, columns=['book', 'word'])
words.head()

# remove empty string
words = words[words.word.str.len() > 0]
words.head()

# calculate TF-IDF statistic- normalize the words by bringing them all to the same case
words['word'] = words.word.str.lower()
words.head()


# counts of the terms per book:
counts = words.groupby('book')\
    .word.value_counts()\
    .to_frame()\
    .rename(columns={'word':'n_w'})
counts.head()


def pretty_plot_top_n(series, top_n=5, index_level=0):
    r = series\
    .groupby(level=index_level)\
    .nlargest(top_n)\
    .reset_index(level=index_level, drop=True)
    r.plot.bar()
    return r.to_frame()


pretty_plot_top_n(counts['n_w'])

word_sum = counts.groupby(level=0)\
    .sum().rename(columns={'n_w': 'n_d'})

tf = counts.join(word_sum)

tf['tf'] = tf.n_w/tf.n_d

tf.head()

# display(pretty_plot_top_n(tf['tf']))
pretty_plot_top_n(tf['tf'])

# get the size of set of unique elements in a series.
c_d = words.book.nunique()

# sorting values is only for the presentation and it is not needed for the further computations.
idf = words.groupby('word')\
    .book\
    .nunique()\
    .to_frame()\
    .rename(columns={'book': 'i_d'})\
    .sort_values('i_d')
idf.head()

idf['idf'] = np.log(c_d/idf.i_d.values)

idf.head()

tf_idf = tf.join(idf)
tf_idf.head()
tf_idf['tf_idf'] = tf_idf.tf * tf_idf.idf
tf_idf.head()

pretty_plot_top_n(tf_idf['tf_idf'])

r = words[words.word.str.match('^s')]\
    .groupby('word')\
    .count()\
    .rename(columns={'book': 'n'})\
    .nlargest(10, 'n')
r.plot.bar()
