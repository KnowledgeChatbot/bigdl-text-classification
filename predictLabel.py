#usr/bin/python
# -*- encoding: utf-8 -*-
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import io
import itertools
import re
import json
import jieba
from pyfasttext import FastText
from bigdl.dataset import news20
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample
from flask import Flask, request,make_response
from flask_restful import Resource, Api
from flask_jsonpify import jsonify


def text_to_words(review_text):
    words = jieba.cut(review_text.replace("\t"," ").replace("\n"," "))
#    uncomment this to check jieba tokenize result.
#    with io.open('join.txt', 'w+',encoding='utf-8') as f:
#        f.write(' '.join(words))
    return words


def analyze_texts(data_rdd):
    def index(w_c_i):
        ((w, c), i) = w_c_i
        return (w, (i + 1, c))
    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0])) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda w_c: - w_c[1]).zipWithIndex() \
        .map(lambda w_c_i: index(w_c_i)).collect()


# pad([1, 2, 3, 4, 5], 0, 6)
def pad(l, fill_value, width):
    if len(l) >= width:
        return l[0: width]
    else:
        l.extend([fill_value] * (width - len(l)))
        return l


def to_vec(token, b_w2v, embedding_dim):
    if token in b_w2v:
        return b_w2v[token]
    else:
        return pad([], 0, embedding_dim)


def to_sample(vectors, label, embedding_dim):
    # flatten nested list
    flatten_features = list(itertools.chain(*vectors))
    features = np.array(flatten_features, dtype='float').reshape(
        [sequence_len, embedding_dim])

    return Sample.from_ndarray(features, np.array(label))


def predict(targetSentence,targetRange,model):
    # tests is an array of tuple (words, label) 
    texts = [(targetSentence,0)] 
    data_rdd = sc.parallelize(texts, 1)

    word_to_ic = analyze_texts(data_rdd)

    # Only take the top wc between [10, sequence_len]
    word_to_ic = dict(word_to_ic)
    bword_to_ic = sc.broadcast(word_to_ic) 
    # word2vec model is the pre-trained FastText model for chinese, since glove     # does not support chinese
    filtered_w2v = dict((w,w2v[w]) for w in w2v.words if w in word_to_ic)
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    tokens_rdd = data_rdd.map(lambda text_label:
                              ([w for w in text_to_words(text_label[0]) if
                                w in bword_to_ic.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(
        lambda tokens_label: (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                       ([to_vec(w, bfiltered_w2v.value,
                                                embedding_dim) for w in
                                         tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], embedding_dim))
    
    #load pre-trained model
    result = model.predict(sample_rdd)
    labels = result.flatMap(lambda text: text)\
                   .zipWithIndex().sortBy(lambda rate: abs(rate[0]))\
                   .map(lambda k:(categories[k[1]],str(abs(k[0])))).collect()
    selectedLabels = labels[0:targetRange]
    return selectedLabels

def initCategories():
    cat = []
    for dirName in os.listdir('/home/azureuser/dump'):
        cat.append(dirName)
    return sorted(cat)
app = Flask(__name__)
sc = SparkContext(appName="text_classifier",
                          conf=create_spark_conf())
faqmodel = Model.loadModel('faqmodel.bigdl','faqmodel.bin')
kbmodel = Model.loadModel('model.bigdl','model.bin')
w2v = FastText('/home/azureuser/cc.zh.300.bin')

max_words = 5000
sequence_len = 500
embedding_dim = 300
expectRange = 3
init_engine()
categories = initCategories()
@app.route("/predict/faq/<targetStr>/<int:expectRange>", methods=['GET'])
def predictFaqResult(targetStr,expectRange):
    result=  predict(targetStr,expectRange,faqmodel)
    return json.dumps(result)

@app.route("/predict/kb/<targetStr>/<int:expectRange>", methods=['GET'])
def predictKbResult(targetStr,expectRange):
    result=  predict(targetStr,expectRange,kbmodel)
    return json.dumps(result)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug = True)
    

