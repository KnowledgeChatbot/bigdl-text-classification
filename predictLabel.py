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
from optparse import OptionParser

import jieba
from pyfasttext import FastText
from bigdl.dataset import news20
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.util.common import Sample
import datetime as dt


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


def build_model(class_num):
    model = Sequential()

    if model_type.lower() == "cnn":
        model.add(TemporalConvolution(embedding_dim, 256, 5)) \
            .add(ReLU()) \
            .add(TemporalMaxPooling(sequence_len - 5 + 1)) \
            .add(Squeeze(2))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 256, p)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 256, p)))
        model.add(Select(2, -1))

    model.add(Linear(256, 128)) \
        .add(Dropout(0.2)) \
        .add(ReLU()) \
        .add(Linear(128, class_num)) \
        .add(LogSoftMax())

    return model


def predict(sc,targetSentence,targetRange,sequence_len, max_words, embedding_dim):
    # tests is an array of tuple (words, label) 
    categories = [\
    'active directory','api management','app service','application gateway','backup','classic','cloud services',\
    'Computer-vision','cosmos db','Emotion','event hubs','expressroute','icp','iot suite','linux','power bi-workspace-collections',\
    'redis cache','service bus-messaging','service bus-relay','service health','site recovery','sql database','sql data-warehouse','virtual machine-scale-sets',\
    'virtual network','vpn gateway','windows','计费、订阅和发票','一般问题','执行与维护','注册问题']
    texts = [(targetSentence,0)] 
    data_rdd = sc.parallelize(texts, 1)

    word_to_ic = analyze_texts(data_rdd)

    # Only take the top wc between [10, sequence_len]
    word_to_ic = dict(word_to_ic)
    bword_to_ic = sc.broadcast(word_to_ic) 
    # word2vec model is the pre-trained FastText model for chinese, since glove     # does not support chinese
    w2v = FastText('/home/azureuser/cc.zh.300.bin')
    filtered_w2v = dict((w, w2v[w]) for w in w2v.words if w in word_to_ic)
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
    model = Model.loadModel('model.bigdl','model.bin')
    result = model.predict(sample_rdd)
    labels = result.flatMap(lambda text: text)\
                   .zipWithIndex().sortBy(lambda rate: abs(rate[0]))\
                   .map(lambda k:(categories[k[1]],abs(k[0]))).collect()
    selectedLabels = labels[0:targetRange]
    for label in selectedLabels:
        print label[0]
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-t", "--target", dest="target")
    parser.add_option("-r", "--range", dest="expectRange", default="3")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="300")  # noqa

    (options, args) = parser.parse_args(sys.argv)
    sc = SparkContext(appName="text_classifier",
                          conf=create_spark_conf())
    target = options.target
    expectRange = int(options.expectRange)
    embedding_dim = int(options.embedding_dim)
    sequence_len = 500
    max_words = 5000

    init_engine()
    predict(sc,target,expectRange,sequence_len, max_words, embedding_dim)
    sc.stop()
    

