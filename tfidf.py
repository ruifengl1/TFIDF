import nltk
from nltk.stem.porter import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import xml.etree.cElementTree as ET
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import zipfile
import os

PARTIALS = False

def gettext(xmltext) -> str:
    """
    Parse xmltext and return the text from <title> and <text> tags
    """
    elements = []
    xmltext = xmltext.encode('ascii', 'ignore') # ensure there are no weird char
    tree = ET.ElementTree(ET.fromstring(xmltext))
    for elem in tree.iterfind('.//title'):
        elements.append(elem.text)
    for elem in tree.iterfind('.//text/*'):
        elements.append(elem.text)
    return ' '.join(elements)


def tokenize(text) -> list:
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3, strip digits.
    """
    tokenized_words = list()
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2]  # ignore a, an, to, at, be, ...
    for word in tokens:
        if word not in ENGLISH_STOP_WORDS:
            tokenized_words.append(word)
    return tokenized_words



def stemwords(words) -> list:
    """
    Given a list of tokens/words, return a new list with each word
    stemmed using a PorterStemmer.
    """
    new_words = []
    ps = PorterStemmer()
    for word in words:
        new_words.append(ps.stem(word))
    return new_words


def tokenizer(text) -> list:
    return stemwords(tokenize(text))


def compute_tfidf(corpus:dict) -> TfidfVectorizer:
    """
    Create and return a TfidfVectorizer object after training it on
    the list of articles pulled from the corpus dictionary. Meaning,
    call fit() on the list of document strings, which figures out
    all the inverse document frequencies (IDF) for use later by
    the transform() function. The corpus argument is a dictionary
    mapping file name to xml text.
    """
    tfidf = TfidfVectorizer(input='content',
                            analyzer='word',
                            preprocessor=gettext,
                            tokenizer=tokenizer,
                            stop_words='english', # even more stop words
                            decode_error = 'ignore')
    return tfidf.fit(list(corpus.values()))


def summarize(tfidf:TfidfVectorizer, text:str, n:int):
    """
    Given a trained TfidfVectorizer object and some XML text, return
    up to n (word,score) pairs in a list. Discard any terms with
    scores < 0.09. Sort the (word,score) pairs by TFIDF score in reverse order.
    """
    results = list()
    matrix = tfidf.transform([text])
    feature_names = tfidf.get_feature_names()
    scores = matrix.toarray()[0]
    word_index = matrix.nonzero()[1]
    for index in word_index:
        if scores[index] >= 0.09:
            results.append((feature_names[index], round(scores[index],3)))
    return sorted(results, key=lambda item: (item[1], item[0]), reverse=True)[:n]



def load_corpus(zipfilename:str) -> dict:
    """
    Given a zip file containing root directory reuters-vol1-disk1-subset
    and a bunch of *.xml files, read them from the zip file into
    a dictionary of (filename,xmltext) associations. Use namelist() from
    ZipFile object to get list of xml files in that zip file.
    Convert filename reuters-vol1-disk1-subset/foo.xml to foo.xml
    as the keys in the dictionary. The values in the dictionary are the
    raw XML text from the various files.
    """
    xml_dict = dict()
    absfilepath = os.path.expanduser(zipfilename)
    with zipfile.ZipFile(absfilepath, 'r') as zipObj:
        # Get list of files names in zip
        listOfiles = zipObj.namelist()
        # Iterate over the list of file names and extract content
        for filename in listOfiles:
            if filename.endswith('.xml'):
                xml_dict[os.path.basename(filename)] = zipObj.read(filename).strip()
    return xml_dict