from tfidf import *
import sys

def getdata():
    if len(sys.argv)==1: # if no file given, read from stdin
        data = sys.stdin.read()
    else:
        f = open(sys.argv[1], "r")
        data = f.read()
        f.close()
    return data.strip()

def main():
    xmltext = getdata()
    text = gettext(xmltext)
    tokenize_words = tokenize(text)
    stemmed_words = stemwords(tokenize_words)
    counts = Counter(stemmed_words)
    ordered_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    for word, count in ordered_counts[:10]:
        print(word, count)

if __name__ == '__main__':
    main()
