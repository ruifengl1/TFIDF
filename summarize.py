import sys
from tfidf import *


def main():

    zipfilename = sys.argv[1]
    summarizefile = sys.argv[2]

    corpus = load_corpus(zipfilename)
    tfidf = compute_tfidf(corpus)
    summarized_scores = summarize(tfidf, corpus[summarizefile], 20)
    for k, v in summarized_scores:
        print(k, v)

if __name__ == '__main__':
    main()
