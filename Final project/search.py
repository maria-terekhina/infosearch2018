from flask import Flask, render_template, request, redirect, url_for
import json
import pickle
import pymorphy2
from judicial_splitter import splitter as sp
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os
from math import log
from gensim import matutils
import numpy as np
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec

app = Flask(__name__)

def preprocessing(input_text, del_stopwords=True, del_digit=True):
    """
    :input: raw text
        1. lowercase, del punctuation, tokenize
        2. normal form
        3. del stopwords
        4. del digits
    :return: lemmas
    """
    russian_stopwords = set(stopwords.words('russian'))
    words = [x.lower().strip(string.punctuation + '»«–…') for x in word_tokenize(input_text)]
    lemmas = [morph.parse(x)[0].normal_form for x in words if x]

    lemmas_arr = []
    for lemma in lemmas:
        if del_stopwords:
            if lemma in russian_stopwords:
                continue
        if del_digit:
            if lemma.isdigit():
                continue
        lemmas_arr.append(lemma)
    return lemmas_arr


class IterDocs(object):
    def __init__(self, text=False, lemmas=False, tagged=False):
        self.text = text
        self.lemmas = lemmas
        self.tagged = tagged

    def __iter__(self):
        for root, dirs, files in os.walk('./avito_parsed'):
            for i, file in enumerate(files):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    if self.tagged is True:
                        yield TaggedDocument(words=json.load(f), tags=[i])
                    elif self.text is True:
                        yield ' '.join(json.load(f))
                    elif self.lemmas is True:
                        yield json.load(f)

def get_d2v_vectors(words):
    '''
    Compute d2v vector for doc
    :param words: list: lemmas of the doc for which to compute d2v vector
    :return: list: d2v vector
    '''
    vec = model_d2v.infer_vector(words)
    return vec


def get_w2v_vectors(doc):
    """
    Get doc w2v vector
    :param doc: str: doc for which to compute vector
    :return: list: cpmputed vector
    """
    all_vect = list()
    for word in doc:
        try:
            all_vect.append(model_w2v.wv[word])
        except:
            continue

    if len(all_vect) != 0:
        d_vect = np.mean(np.array(all_vect), axis=0)
    else:
        d_vect = [0] * 300
    return d_vect

def score_BM25(qf, dl, avgdl, k1, b, N, n) -> float:
    """
    Compute similarity score between search query and documents from collection
    :return: score
    """
    idf = log((N - n + 0.5) / (n + 0.5))
    score = (idf * (k1 + 1) * qf) / (qf + k1 * (1 - b + b * dl / avgdl))

    return score


def compute_sim(query, doc, inv_index, k1, b, avgdl, N) -> float:
    """
    Compute parameters for BM25 score and pass them to the calculation function
    :param query: str: word for which to claculate BM25
    :param doc: str: doc for which to claculate BM25
    :param inv_index: default_dict: inverted index for the collection, that includes doc
    :return: score
    """
    qf = doc.count(query)
    dl = len(doc)

    if query in inv_index:
        n = len(inv_index[query])
    else:
        n = 0

    return score_BM25(qf, dl, avgdl, k1, b, N, n)


def search_inv(query, corpus, inv_index) -> list:
    """
    Search documents relative to query using inverted index algorithm.
    :param query: str: input text
    :param questions: list: all questions from corpus
    :param answers: list: all answers from corpus
    :param inv_index: list: questions inverted index
    :return: list: 5 relevant answers
    """

    def mean(numbers):
        return float(sum(numbers)) / max(len(numbers), 1)

    k1 = 2.0
    b = 0.75
    file_lens = [len(file) for file in IterDocs(lemmas=True)]
    avgdl = mean(file_lens)
    N = len(file_lens)

    query_list = preprocessing(query)
    scores = list()

    for i, doc in enumerate(corpus):
        score = 0
        for word in query_list:
            score += compute_sim(word, doc, inv_index, k1, b, avgdl, N)
        scores.append([i, score])

    ranked = sorted(scores, key=lambda x: x[1], reverse=True)

    result = list()
    names = list()
    i = 0
    while len(result) < 5:
        doc = ranked[i]
        name = os.listdir('./avito_parsed')[doc[0]][:-7]
        if name[-1] is '_':
            name = name[:-1]
        name += '.txt'

        if not name in names:
            names.append(name)
            with open('./avito_texts/%s' % (name), 'r', encoding='utf-8') as f:
                result.append(f.read())
        i += 1

    return result

def similarity(v1, v2):
    '''
    Compute cosine similarity for 2 vectors
    :param v1, v2: list: vectors
    :return: float: vectors' similarity
    '''
    v1_norm = matutils.unitvec(np.array(v1))
    v2_norm = matutils.unitvec(np.array(v2))
    sim = np.dot(v1_norm, v2_norm)
    if sim is not None:
        return sim
    else:
        return 0

def search_w2v(query, w2v_base) -> list:
    """
    Search documents relative to query using inverted w2v algorithm.
    :param query: str: input text
    :param w2v_base_quest: list: all questions' vectors from corpus
    :param answers: list: all answers from corpus
    :return: list: 5 relative answers
    """

    similarities = list()

    for part in sp(query, 3):
        lemmas = preprocessing(query)
        vec = get_w2v_vectors(lemmas)

        for doc in w2v_base:
            s = similarity(vec, doc['vec'])
            similarities.append({'id': doc['id'], 'sim': s})

    ranked = sorted(similarities, key=lambda x: x['sim'], reverse=True)

    result = list()
    names = list()
    i = 0
    while len(result) < 5:
        doc = ranked[i]
        name = doc['id'][:-7]
        if name[-1] is '_':
            name = name[:-1]
        name += '.txt'

        if not name in names:
            names.append(name)
            with open('./avito_texts/%s' % (name), 'r', encoding='utf-8') as f:
                result.append(f.read())
        i += 1

    return result


def search_d2v(query, d2v_base) -> list:
    """
    Search documents relative to query using inverted d2v algorithm.
    :param query: str: input text
    :param d2v_base_quest: list: all questions' vectors from corpus
    :param answers: list: all answers from corpus
    :return: list: 5 relative answers
    """
    similarities = list()

    for part in sp(query, 3):
        lemmas = preprocessing(query)
        vec = get_d2v_vectors(lemmas)

        for doc in d2v_base:
            s = similarity(vec, doc['vect'])
            similarities.append({'id': doc['id'], 'sim': s})

    ranked = sorted(similarities, key=lambda x: x['sim'], reverse=True)

    result = list()
    names = list()
    i = 0
    while len(result) < 5:
        doc = ranked[i]
        name = doc['id'][:-7]
        if name[-1] is '_':
            name = name[:-1]
        name += '.txt'

        if not name in names:
            names.append(name)
            with open('./avito_texts/%s' % (name), 'r', encoding='utf-8') as f:
                result.append(f.read())
        i += 1

    return result

def search(query, search_method):
    if search_method == 'inverted_index':
        search_result = search_inv(query, IterDocs(lemmas=True), inv_base)
    elif search_method == 'word2vec':
        search_result = search_w2v(query, w2v_base)
    elif search_method == 'doc2vec':
        search_result = search_d2v(query, d2v_base)
    else:
        raise TypeError('unsupported search method')
    return search_result


@app.route('/')
def search_fucntion():
    if request.args:
        print(request.args)
        query = request.args['query']
        method = request.args['method']
        res = search(query, search_method=method)

        output = list()
        for r in res:
            sent = r.split('\n')
            output.append([sent[0], ' '.join(sent[1:])[:200] + '...'])
        return render_template('results.html', results=output, query=query)
    else:
        return render_template('index.html')


def load_models():
    morph = pymorphy2.MorphAnalyzer()
    print('morph')

    model_w2v = Word2Vec.load(
        r'araneum_none_fasttextskipgram_300_5_2018/araneum_none_fasttextskipgram_300_5_2018.model')
    model_d2v = Doc2Vec.load(r'doc2vec_model_avito.model')

    with open('inv_base.pkl', 'rb') as f:
        inv_base = pickle.load(f)
    with open('w2v_base.pkl', 'rb') as f:
        w2v_base = pickle.load(f)
    with open('d2v_base.pkl', 'rb') as f:
        d2v_base = pickle.load(f)
    print('model')
    return morph, model_w2v, model_d2v, inv_base, w2v_base, d2v_base

if __name__ == '__main__':
    morph, model_w2v, model_d2v, inv_base, w2v_base, d2v_base = load_models()
    app.run(use_reloader=False, debug=True)


