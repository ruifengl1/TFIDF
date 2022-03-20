# TFIDF

## Objective

The goal of this project is to explore a core technique used in text analysis called TFIDF or term frequency, inverse document frequency. We will use what is called a bag-of-words representation where the order of words in a document doesn't matter--we care only about the words and how often they occur. A word's TFIDF value is often used as a feature for document clustering or classification. The more a term helps to distinguish its enclosing document from other documents, the higher its TFIDF score. As such, words with high TFIDF scores are often very good summarizing keywords for document.

## Dataset

- reuters-vol1-disk1-subset.zip

The articles (text files) in the zip file look like this fictitious file's contents:

```xml
<?xml version="1.0" encoding="iso-8859-1" ?>
<newsitem itemid="99" id="root" date="1996-10-21" xml:lang="en">
<title>Cats Do Hip Hop</title>
<dateline>USA 1996-10-21</dateline>
<text>
<p>Check this out.</p>
<p>Hilarious.</p>
</text>
<link>http://www.huffingtonpost.co.uk/2014/06/06/kittens-dance-turn-it-down-for-what_n_5458093.html</link>
<metadata>
<codes class="bip:countries:1.0">
  <code code="USA">
    <editdetail attribution="Cat Reuters BIP Coding Group" action="confirmed" date="1996-10-21"/>
  </code>
</codes>
<dc element="dc.date.created" value="1996-10-21"/>
<dc element="dc.source" value="Cat Reuters"/>
</metadata>
</newsitem>
```        

## Commandline
In `common.py` , it will summarize an article by showing the most common 10 words.  
```
$ python common.py ~/data/reuters-vol1-disk1-subset/33313newsML.xml
gener 19
power 14
transmiss 14
new 12
said 12
electr 11
cost 10
zealand 9
signal 8
tran 7
```

The application use TFIDF on a corpus of articles from which we can compute the term frequency across articles.  Here is how we will execute our program (`summarize.py`):

```bash
$ python summarize.py ~/data/reuters-vol1-disk1-subset.zip 33313newsML.xml
transmiss 0.428
gener 0.274
power 0.254
electr 0.253
zealand 0.235
tran 0.215
signal 0.214
esanz 0.191
cost 0.162
leay 0.143
gisborn 0.143
charg 0.131
new 0.130
island 0.128
auckland 0.113
effici 0.110
pricipl 0.096
eastland 0.096
```

So, we pass in the overall corpus (**as a zip file**) and then a specific file for which we want the top TFIDF scored words. The output shows max 20 words and with **three decimals of precision**. Print only those words scoring >= 0.09.