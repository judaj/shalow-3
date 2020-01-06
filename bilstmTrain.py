import sys
import numpy as np
import random as rand
import dynet as dy
import pickle
import time


class DataParser(object):
    def __init__(self):
        self.vocab = set()
        self.tags = set()
        self.pref = set()
        self.suff = set()
        self.data = []
        self.chars = set()
        # self.vocab.add(UNK)
        # self.pref.add(UNK)
        # self.suff.add(UNK)
        # self.chars.add(UNK)

    def parse(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        sentence = []
        for idx, line in enumerate(lines):
            line = line.strip()
            if line == '':
                self.data.append(sentence)
                sentence = []
                continue
            word, tag = line.split()
            word = word.lower()
            # print(UNK)
            # print(word)
            sentence.append((word, tag))
            self.suff.add(word[-SUFF_SIZE:])
            self.pref.add(word[:PREF_SIZE])
            self.vocab.add(word)
            self.tags.add(tag)
            for char in word:
                self.chars.add(char)


class BilstmTagger(object):
    def __init__(self, data, rper):
        self.data = data
        self.rper = rper
        self.model = dy.ParameterCollection()
        # backward and forward LSTMs
        # 2nd layer with HIDDEN * 2 as we concatenate
        self.lstm_f_1 = dy.LSTMBuilder(LYER_SIZE, DIM_EMBD, DIM_HID, self.model)
        self.lstm_f_2 = dy.LSTMBuilder(LYER_SIZE, DIM_HID * 2, DIM_HID, self.model)
        self.lstm_b_1 = dy.LSTMBuilder(LYER_SIZE, DIM_EMBD, DIM_HID, self.model)
        self.lstm_b_2 = dy.LSTMBuilder(LYER_SIZE, DIM_HID * 2, DIM_HID, self.model)
        # parameters
        self.embed_word = self.model.add_lookup_parameters((len(data.vocab) + 1, DIM_EMBD))
        self.embed_lstm = dy.LSTMBuilder(LYER_SIZE, DIM_EMBD, DIM_EMBD, self.model)
        self.embed_char = self.model.add_lookup_parameters((len(data.chars) + 1, DIM_EMBD))
        self.embed_pref = self.model.add_lookup_parameters((len(data.pref) + 1, DIM_EMBD))
        self.embed_suff = self.model.add_lookup_parameters((len(data.suff) + 1, DIM_EMBD))
        self.embed_linear = self.model.add_parameters((DIM_EMBD, DIM_EMBD * 2))
        self.output = self.model.add_parameters((len(data.tags), DIM_HID * 2))
        #trainer
        self.trainer = dy.AdamTrainer(self.model)

        # dicts
        self.w2i = {w: i for i, w in enumerate(data.vocab)}
        # self.int2word = {i: w for i, w in enumerate(data.vocab)}
        self.t2i = {t: i for i, t in enumerate(data.tags)}
        self.i2t = {i: t for i, t in enumerate(data.tags)}
        self.p2i = {p: i for i, p in enumerate(data.pref)}
        # self.int2pref = {i: p for i, p in enumerate(data.pref)}
        self.s2i = {s: i for i, s in enumerate(data.suff)}
        # self.int2suff = {i: s for i, s in enumerate(data.suff)}
        self.c2i = {c: i for i, c in enumerate(data.chars)}
        # self.int2char = {i: c for i, c in enumerate(data.chars)}

    def train_net(self, dev):
        for epoch_index in range(EPOC_NUM):
            train_data = self.data.data
            rand.shuffle(train_data)
            # print(len(train_data))
            for index, sent in enumerate(train_data, 1):
                words_list, tags_list = [word for word, tag in sent], [tag for word, tag in sent]
                tags_list = [self.t2i.get(tag, len(self.data.tags)) for tag in tags_list]
                output_pred_list = self.predict_data(words_list)
                self.loss(output_pred_list, tags_list).backward()
                self.trainer.update()
                if index % 500 == 0:
                    a = str(1+epoch_index) + ' ' + str(index) + ', DevAcc=' + str(self.accuracy_calc(dev) * 100) + '%.' + "\n"
                    fru.write(a)
                    print(str(1+epoch_index) + ' ' + str(index) + ', DevAcc=' + str(self.accuracy_calc(dev) * 100) + '%.')

    def word_embedding_d(self, word):
        chars = [self.embed_char[self.c2i.get(c, len(self.c2i))] for c in word]
        chars_out = self.embed_lstm.initial_state().transduce(chars)[-1]
        return dy.tanh(self.embed_linear * dy.concatenate([self.embed_word[self.w2i.get(word, len(self.w2i))], chars_out]))

    def word_embedding_c(self, word):
        word_emb = self.embed_word[self.w2i.get(word, len(self.w2i))]
        pref_emb = self.embed_pref[self.p2i.get(word[:PREF_SIZE], len(self.p2i))]
        suff_emb = self.embed_suff[self.s2i.get(word[-SUFF_SIZE:], len(self.s2i))]
        return dy.esum([word_emb, pref_emb, suff_emb])

    def word_embedding_b(self, word):
        chars = [self.embed_char[self.c2i.get(c, len(self.c2i))] for c in word]
        return self.embed_lstm.initial_state().transduce(chars)[-1]

    def word_embedding_a(self, word):
        return self.embed_word[self.w2i.get(word, len(self.w2i))]

    def word_embedding(self, word):
        if self.rper == 'a':
            return self.word_embedding_a(word)
        elif self.rper == 'b':
            return self.word_embedding_b(word)
        elif self.rper == 'c':
            return self.word_embedding_c(word)
        elif self.rper == 'd':
            return self.word_embedding_d(word)
        else:
            raise ValueError()

    def predict_data(self, x, val=False):
        dy.renew_cg()
        x = [self.word_embedding(xi) for xi in x]
        lstm_f1_s = self.lstm_f_1.initial_state()
        lstm_b1_s = self.lstm_b_1.initial_state()
        out_f1 = lstm_f1_s.transduce(x)
        out_b1 = lstm_b1_s.transduce(reversed(x))
        out_layer1 = [dy.concatenate([of, ob]) for of, ob in zip(out_f1, reversed(out_b1))]

        lstm_f2_s = self.lstm_f_2.initial_state()
        lstm_b2_s = self.lstm_b_2.initial_state()
        out_f2 = lstm_f2_s.transduce(out_layer1)
        out_b2 = lstm_b2_s.transduce(reversed(out_layer1))
        out_layer2 = [dy.concatenate([of, ob]) for of, ob in zip(out_f2, reversed(out_b2))]
        outputes = []
        if not val:
            for ou in out_layer2:
                outputes.append(self.output * ou)
            return outputes
        tags = []
        for ou in out_layer2:
            tags.append(self.i2t[np.argmax(dy.softmax(self.output * ou).npvalue())])
        return tags

    def accuracy_calc(self, data):
        good = 0
        total = 0
        data = data.data
        words, tags = [[word for word, tag in sentance] for sentance in data], [[tag for word, tag in sentance] for sentance in data]
        predicts_tags = [self.predict_data(x, val=True) for x in words]

        for y_s, y_hat_s in zip(tags, predicts_tags):
            for y, y_hat in zip(y_s, y_hat_s):
                total += 1
                if y == y_hat:
                    good += 1

        return good / total

    def load(self, model_fname):
        self.model.populate(model_fname)

    def save_data_and_modle(self, model_fname, data_for_train):
        pickle.dump(data_for_train, open("data_" + model_fname +"_" + self.rper, "wb"))
        model_fname = model_fname + self.rper
        self.model.save(model_fname)

    @staticmethod
    def loss(output_pred, real_tags):
        all_loss = []
        for output, tag in zip(output_pred, real_tags):
            all_loss.append(dy.pickneglogsoftmax(output, tag))
        return dy.esum(all_loss)

LYER_SIZE = 1
DIM_EMBD = 60
DIM_HID = 80
EPOC_NUM = 5
PREF_SIZE = 3
SUFF_SIZE = 3


if __name__ == "__main__":

    rper, train_fname, model_fname, dev_fname = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    fru = open(str(rper)+"_acc.txt", "w")
    train_data = DataParser()
    dev_data = DataParser()
    train_data.parse(train_fname)
    dev_data.parse(dev_fname)
    start = time.time()
    bilstm = BilstmTagger(train_data, rper)
    bilstm.train_net(dev_data)
    bilstm.save_data_and_modle(model_fname, train_data)

    end = time.time()
    print('run:' + str(end - start) + ' seconds')