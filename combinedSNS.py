import numpy as np
import pandas as pd
import requests
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from janome.tokenizer import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer as kerastokenizer
from tensorflow.keras.models import load_model

url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
response = requests.get(url)
stopwords = [w for w in response.content.decode().split('\r\n') if w != '']

char_filters = [UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(u'蛇の目', u'janome')]
tokenizer = Tokenizer()
token_filters = [CompoundNounFilter(), POSStopFilter(['記号', '助詞']), LowerCaseFilter()]
a = Analyzer(char_filters=char_filters, tokenizer=tokenizer, token_filters=token_filters)


def cleantext(text):
    text = ' '.join(text)
    return re.sub(' +', ' ', re.sub(
        r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s(
        )<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
        " ", text).replace('https://', ''))


def fetch_tokenize(text):
    tokenized_text = []
    for chunk in [token.surface for token in a.analyze(text)]:
        if chunk not in stopwords:
            tokenized_text.append(chunk.lower())
    return cleantext(' '.join(tokenized_text))


def preprocessing(text):
    dict = {}
    dict['tokenized_comment'] = [fetch_tokenize(text)]
    df = pd.DataFrame(dict)
    tokenizer = kerastokenizer(split=' ')
    tokenizer.fit_on_texts(df)
    df = tokenizer.texts_to_sequences(df)
    df = pad_sequences(df, maxlen=1301)
    predict = prediction(df)
    return predict


def prediction(X_test):
    # X_test = np.array(pd.read_csv('X_test.csv'))
    # Y_test = np.array(pd.read_csv('Y_test.csv'))
    prediction = ''
    for i in X_test:
        w, q = complain(i)
        # print(q)
        if q == 1:
            prediction = 'Opinion'
        elif q == 0:
            ar, lab = bug(i)
            # print(max(ar[0]), lab)
            if max(ar[0]) > 0.80 and lab == 0:
                prediction = 'Critical Bug'
            elif max(ar[0]) > 0.80 and lab == 1:
                prediction = 'Non-Critical Bug'
            else:
                prediction = 'Complain'
    # print(prediction)
    return prediction


def complain(i):
    model = load_model("complainopinion")
    w = model.predict(np.array([i, ]))
    q = model.predict_classes(np.array([i, ]))
    return w, q


def bug(i):
    model = load_model("cbncb")
    w = model.predict(np.array([i, ]))
    q = model.predict_classes(np.array([i, ]))
    return w, q

# complain->0, opiniom->1
# 1-.[0,1], 0-[1,0]
# cb-0, ncb-1
# preprocessing('あたり　今月最後かも')
