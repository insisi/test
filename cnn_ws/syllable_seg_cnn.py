# -*- coding: utf-8 -*-
import pickle
import numpy as np
import time
import sys
import codecs
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer.functions.loss import softmax_cross_entropy
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers

class  CharSeg(Chain):
    def __init__(self, char_vocab=60,char_emb_dim=50, char_window_size=5, char_init_emb=None, char_hidden_dim=100, tag_num=2):
        #CRF Layers
        self.crf=L.CRF1d(n_label=2)
        super(CharSeg,self).__init__(
            char_emb=L.EmbedID(char_vocab,char_emb_dim),
            char_conv=L.Convolution2D(1,char_hidden_dim,
                                      ksize=(char_emb_dim,char_window_size),
                                      stride=(1,1),
                                      pad=0),
            predict=L.Linear(char_hidden_dim,tag_num),
        )

    def conv_max(self, char_init_emb):
        char_shape=char_init_emb.data.shape #sent_size x char_size x char_dim
        char_emb=F.reshape(char_init_emb,(char_shape[0],1,char_shape[1],char_shape[2]))
        char_emb_tran=F.transpose(char_emb,(0,1,3,2))
        char_emb_conv=self.char_conv(char_emb_tran)
        sent_size, char_hidden, _, fi_map = char_emb_conv.data.shape
        char_emb_conv_reshape=F.reshape(char_emb_conv,(sent_size,char_hidden,fi_map))
        char_emb_conv_tran=F.transpose(char_emb_conv_reshape,(0,2,1))
        max_pool=F.max(char_emb_conv_tran,axis=1)
        feature_vec=self.predict(F.relu(max_pool, use_cudnn=False))
        return feature_vec


    def expand(self, input_data, char_win_size, pad_len):
        sent = []
        for i in range(len(input_data) - pad_len):
            sent.append(input_data[i:i + char_win_size])
        return sent

    def eval_word_level(self, target):  # to separate sent to word to evaluate, input is a list of sent from data
        sp_target = [[]]
        num = 0
        for i in range(len(target)):
            if target[i] == 0:
                num += 1
                sp_target.append([])
                continue
            sp_target[num].append(target[i])
        sp_target.pop(0)
        for k in range(len(sp_target)):
            sp_target[k].insert(0, 0)
        sp_target = np.array(sp_target)
        return sp_target

    def join_list(self, lst):
        lst = ''.join(map(str, lst))
        return lst

    def count_correct(self, predict, target):
        i = 0
        correct = 0
        prevB = 0
        target += '0'
        predict += '0'
        while i < len(target) - 1:
            if predict[i] != target[i]:
                i = target.find('0', i + 1)
                prevB = i
            else:
                if (target[i + 1] == '0' and predict[i + 1] == '0'):
                    # print prevB, target[prevB:i + 1]
                    correct += 1
                    prevB = i + 1
                i += 1
        # print correct
        return correct

    def cnn_train(self, x_train, y_train, n_epochs, model_name):
        data_size = len(x_train)
        start = time.time()
        print ('start time',start)
        for epoch in range(n_epochs):
            print ('epoch %d'%epoch)
            print ('-----------------------------')
            index = np.random.permutation(data_size)
            sum_loss_train = 0.0
            sum_correct = 0.0
            sum_total = 0.0
            sum_sys_out = 0.0
            for i in range(0, data_size):
                x = x_train[index[i]]
                y = np.array(y_train[index[i]], dtype=np.int32)
                y = Variable(y)
                x_id = cnn_char_seg.expand(x, 5, 4)
                x_id_var = Variable(np.array(x_id, dtype=np.int32))
                char_emb_init = cnn_char_seg.char_emb(x_id_var)
                predict_vec = cnn_char_seg.conv_max(char_emb_init)
                predict = np.argmax(predict_vec.data, axis=1)  # predict vector
                predict_lst = list(predict)  # change from numpy to list
                target_lst = list(y.data)  # chage from numpy to list
                eval_predict = cnn_char_seg.eval_word_level(predict)  # count system output token
                eval_target = cnn_char_seg.eval_word_level(y.data)  # count total token
                total = len(eval_target)
                sys_out = len(eval_predict)
                join_pred = cnn_char_seg.join_list(predict_lst)
                join_tar = cnn_char_seg.join_list(target_lst)
                correct = cnn_char_seg.count_correct(join_pred, join_tar)
                sum_correct += correct
                sum_total += total
                sum_sys_out += sys_out
                opt.zero_grads()
                loss = F.softmax_cross_entropy(predict_vec, y)
                sum_loss_train += loss.data
                loss.backward()
                opt.update()
            print ("loss_train=", float(sum_loss_train) / len(x_train))
            print ("correct sys_out and total", sum_correct, sum_sys_out, sum_total)
            print ("recall_train", float(sum_correct) / sum_total)
            print ("precison_train", float(sum_correct) / sum_sys_out)
        print ("Model Saved")
        serializers.save_npz(model_name, cnn_char_seg)
        end = time.time()
        total = end - start
        print ("total time used", total)

    def cnn_test(self, x_test, y_test, n_epochs, model_name, model):
        serializers.load_npz(model_name, model)
        data_size = len(x_test)
        for epoch in range(n_epochs):
            print ("epoch %d" % epoch)
            print ("------------------------------")
            index = np.random.permutation(data_size)
            sum_loss_test = 0.0
            sum_correct_test = 0.0
            sum_total_test = 0.0
            sum_sys_out = 0.0
            for i in range(0, data_size):
                x = x_test[index[i]]
                # print x
                y = np.array(y_test[index[i]], dtype=np.int32)
                y = Variable(y)
                x_id = cnn_char_seg.expand(x, 5, 4)
                x_id_var = Variable(np.array(x_id, dtype=np.int32))
                char_emb_init = cnn_char_seg.char_emb(x_id_var)
                predict_vec = cnn_char_seg.conv_max(char_emb_init)
                predict = np.argmax(predict_vec.data, axis=1)
                eval_predict = cnn_char_seg.eval_word_level(predict)
                eval_target = cnn_char_seg.eval_word_level(y.data)
                predict_lst = list(predict)
                target_lst = list(y.data)
                total = len(eval_target)
                sys_out = len(eval_predict)
                join_pred = cnn_char_seg.join_list(predict_lst)
                join_tar = cnn_char_seg.join_list(target_lst)
                correct_test = cnn_char_seg.count_correct(join_pred, join_tar)
                sum_correct_test += correct_test
                sum_total_test += total
                sum_sys_out += sys_out
                loss = F.softmax_cross_entropy(predict_vec, y)
                sum_loss_test += loss.data
            print ('loss_test = ', float(sum_loss_test) / len(x_test))
            print ("correct and total", sum_correct_test, sum_total_test)
            print ('recall_test', float(sum_correct_test) / sum_total_test)
            print ("precision_test", float(sum_correct_test) / sum_sys_out)

    def decode_x(self,x_lst,id_dict_path,padding):
        id_dict= pickle.load(open(id_dict_path, 'rb'))
        for i in range(len(x_lst)):
            if x_lst[i] in id_dict:
                x_lst[i] = id_dict[x_lst[i]]
            else:
                x_lst[i] = 'unk'
        for i in range(2):
            x_lst.pop()
            x_lst.pop(0)
        return x_lst

    def decode_syllable(self,x_test, y_test, n_epochs, model_name, model,decode_syllable):
        #sys.stdout = codecs.open(decode_syll, 'w', encoding='utf-8')
        serializers.load_npz(model_name, model)
        data_size = len(x_test)
        syllable_tag=[]
        syll_dict= pickle.load(open('id_char_syllable.dict', 'rb'))
        for epoch in range(n_epochs):
            index = np.random.permutation(data_size)
            #print (index)

            #idx=index = np.random.permutation(len(x_try))
            sum_loss_test = 0.0
            sum_correct_test = 0.0
            sum_total_test = 0.0
            sum_sys_out = 0.0
            for i in range(0, data_size):
                #x_idx=x_try[idx[i]]
                x = x_test[i]
                # # print x
                y = np.array(y_test[i], dtype=np.int32)
                y = Variable(y)
                x_id = cnn_char_seg.expand(x, 5, 4)
                x_id_var = Variable(np.array(x_id, dtype=np.int32))
                char_emb_init = cnn_char_seg.char_emb(x_id_var)
                predict_vec = cnn_char_seg.conv_max(char_emb_init)
                predict = np.argmax(predict_vec.data, axis=1)
                eval_predict = cnn_char_seg.eval_word_level(predict)
                eval_target = cnn_char_seg.eval_word_level(y.data)
                predict_lst = list(predict)
                target_lst = list(y.data)
                total = len(eval_target)
                sys_out = len(eval_predict)
                join_pred = cnn_char_seg.join_list(predict_lst)
                join_tar = cnn_char_seg.join_list(target_lst)
                x_lst=cnn_char_seg.decode_x(x,'id_char_syllable.dict',2)
                syllable_tag.append([x_lst,predict])
        decode_syllable_lst=[]
        syllable=[]
        pair=[]
        for i in range(len(syllable_tag)):
            syllable.append([])
            pair=list(zip(syllable_tag[i][0],syllable_tag[i][1]))
            decode_syllable_lst.append(pair)
        for i in range(len(decode_syllable_lst)):
            for j in range(len(decode_syllable_lst[i])):
                decode_syllable_lst[i][j]=list(decode_syllable_lst[i][j])
        print (decode_syllable_lst)
        combine_sent = []
        for s in decode_syllable_lst:
            w = []
            ws = []
            for c, n in s:
                if n == 0:
                    if w:
                        ws.append("".join(w))
                    w = []
                w.append(c)
            ws.append("".join(w))
            combine_sent.append(ws)
        #pickle.dump(combine_sent, open(decode_syllable, "wb"))
        print (combine_sent)
        return combine_sent

    def decode_data(self,load_file, save_file,syll_dict,new_syllable_dict,new_id_syllable_dict,model_name, model,label_test_1,x_test):
        #sys.stdout = codecs.open(save_file, 'w', encoding='utf-8')
        syllable_dict=pickle.load(open(syll_dict, 'rb'))
        serializers.load_npz(model_name, model)
        sum_loss_test = 0.0
        sum_correct_test = 0.0
        sum_total_test = 0.0
        sum_sys_out = 0.0
        sent = [[]]
        num = 0
        init_char = []
        count = 0
        new_sent = []
        token = [[]]
        new_id_syllable={}
        unk_syll=[]
        token_tag=[]
        for line in codecs.open(load_file, 'r', encoding='utf-8'):
            line = line.strip().split()
            if len(line) != 2:
                sent.append([])
                num+=1
                continue
            sent[num].append([line[0], line[1]])
        combine_sent = []
        for s in sent:
            w = []
            ws = []
            for c, n in s:
                if n == "0":
                    if w:
                        ws.append("".join(w))
                    w = []
                w.append(c)
            ws.append("".join(w))
            combine_sent.append(ws)
        for i in range(len(combine_sent)):
            for j in range(len(combine_sent[i])):
                if combine_sent[i][j] not in syllable_dict:
                    unk_syll.append(combine_sent[i][j])
        for i in range(len(unk_syll)):
            syllable_dict[unk_syll[i]]=len(syllable_dict)+i
        #pickle.dump(syllable_dict, open(new_syllable_dict, "wb"))
        for char, id in syllable_dict.items():
            new_id_syllable[id] = char
        #pickle.dump(new_id_syllable, open(new_id_syllable_dict, "wb"))
        #token_char=token[:]
        for i in range(len(combine_sent)):
            for j in range(len(combine_sent[i])):
                if combine_sent[i][j] in syllable_dict:
                    combine_sent[i][j]=syllable_dict[combine_sent[i][j]]
        for i in range(len(combine_sent)):
            combine_sent_padding=[0,0]+combine_sent[i]+[0,0]
            y = np.array(label_test_1[i], dtype=np.int32)
            y = Variable(y)
            #x=[[0,0,2,31,23,1,0,0]]
            x_id = cnn_char_seg.expand(combine_sent_padding, 5, 4)
            x_id_var = Variable(np.array(x_id, dtype=np.int32))
            char_emb_init = cnn_char_seg.char_emb(x_id_var)
            predict_vec = cnn_char_seg.conv_max(char_emb_init)
            predict = np.argmax(predict_vec.data, axis=1)
            x_input=cnn_char_seg.decode_x(x_test[i],'_new_id_sylable_dict_1.dict',2)
            token_lst=cnn_char_seg.decode_x(combine_sent_padding,'_new_id_sylable_dict_1.dict',2)

        x_try = pickle.load(open('unk_sent_id_2_padding.lst', 'rb'))
        print (x_try[2644:3306])

            # print ('predit',len(predict),predict)
            # print ('targ',len(y.data),y.data)
        #     token_lst=cnn_char_seg.decode_x(combine_sent_padding,'_new_id_sylable_dict_1.dict',2)
        #     token_tag.append([token_lst, predict])
        # for i in range(len(token_tag)):
        #     pair = list(zip(token_tag[i][0], token_tag[i][1]))
        #     for j in pair:
        #         print(j[0], j[1])
        #     print(' ')
            # print (token_lst)

        #     token_tag.append([, predict])
        # for i in range(len(syllable_tag)):
        #     pair = list(zip(syllable_tag[i][0], syllable_tag[i][1]))
        #     for j in pair:
        #         print(j[0], j[1])
        #     print(' ')
        #     eval_predict = cnn_char_seg.eval_word_level(predict)
        #     eval_target = cnn_char_seg.eval_word_level(y.data)
        #     predict_lst = list(predict)
        #     target_lst = list(y.data)
        #     total = len(eval_target)
        #     sys_out = len(eval_predict)
        #     join_pred = cnn_char_seg.join_list(predict_lst)
        #     join_tar = cnn_char_seg.join_list(target_lst)
        #     correct_test = cnn_char_seg.count_correct(join_pred, join_tar)
        #     sum_correct_test += correct_test
        #     sum_total_test += total
        #     sum_sys_out += sys_out
        # print("correct and total", sum_correct_test, sum_total_test)
        # print('recall_test', float(sum_correct_test) / sum_total_test)
        # print("precision_test", float(sum_correct_test) / sum_sys_out)
        # #token_lst=cnn_char_seg.decode_x(token_padding,'new_lao_sylable_dict_1.dict',2)


    def generate_data(self, x_input_file, label_file):  # x_input->sent2id.lst label->label_emb.lst
        x_input = pickle.load(open(x_input_file, 'rb'))
        label = pickle.load(open(label_file, 'rb'))
        x_set_1 = x_input[0:661]
        label_set_1 = label[0:661]
        x_set_2 = x_input[661:1322]
        label_set_2 = label[661:1322]
        x_set_3 = x_input[1322:1983]
        label_set_3 = label[1322:1983]
        x_set_4 = x_input[1983:2644]
        label_set_4 = label[1983:2644]
        x_set_5 = x_input[2644:3306]
        label_set_5 = label[2644:3306]

        x_train_set_1 = x_set_1 + x_set_2 + x_set_3+x_set_4
        x_test_set_1 = x_set_5

        label_train_set_1 = label_set_1 + label_set_2 + label_set_3+label_set_4
        label_test_set_1 = label_set_5

        x_train_set_2 = x_set_2 + x_set_3 + x_set_4+x_set_5
        x_test_set_2 = x_set_1

        label_train_set_2 = label_set_2 + label_set_3 + label_set_4+label_set_5
        label_test_set_2 = label_set_1

        x_train_set_3 = x_set_3 + x_set_4 + x_set_5+x_set_1
        x_test_set_3 = x_set_2

        label_train_set_3 = label_set_3 + label_set_4 + label_set_5+label_set_1
        label_test_set_3 = label_set_2

        x_train_set_4 = x_set_4 + x_set_5 + x_set_1+x_set_2
        x_test_set_4 = x_set_3

        label_train_set_4 = label_set_4 + label_set_5 + label_set_1+label_set_2
        label_test_set_4 = label_set_3

        x_train_set_5 = x_set_5 + x_set_1 + x_set_2 +x_set_3
        x_test_set_5 = x_set_4

        label_train_set_5 = label_set_5 + label_set_1 + label_set_2+label_set_3
        label_test_set_5 = label_set_4

        return x_train_set_1, x_test_set_1, label_train_set_1, label_test_set_1, \
               x_train_set_2, x_test_set_2, label_train_set_2, label_test_set_2, \
               x_train_set_3, x_test_set_3, label_train_set_3, label_test_set_3, \
               x_train_set_4, x_test_set_4, label_train_set_4, label_test_set_4, \
               x_train_set_5, x_test_set_5, label_train_set_5, label_test_set_5

sent_id_path='char_sent_id_2_padding.lst'
label_path='char_syllable_lable_emb.lst'
model_path='/Users/cl-lab/Desktop/cnn_lao_seg/file/model/new_cnn_train_5.model'

cnn_char_seg = CharSeg(char_vocab=60,char_emb_dim=50, char_window_size=5,char_hidden_dim=100, tag_num=2)
opt = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
opt.setup(cnn_char_seg)
x_train_1,x_test_1,label_train_1,label_test_1,\
x_train_2,x_test_2,label_train_2,label_test_2,\
x_train_3,x_test_3,label_train_3,label_test_3,\
x_train_4,x_test_4,label_train_4,label_test_4,\
x_train_5,x_test_5,label_train_5,label_test_5=cnn_char_seg.generate_data(sent_id_path,label_path)

x_input = pickle.load(open('id_word_pos_combine.dict', 'rb'))
print (x_input)
#model_1=cnn_char_seg.cnn_train(x_train_1,label_train_1, 100, 'ws_cnn_train_1.model')
#model_2=cnn_char_seg.cnn_train(x_train_2,label_train_2, x_dev_2,label_dev_2, 100, 'syllable_cnn_train_2.model')
#model_3=cnn_char_seg.cnn_train(x_train_3,label_train_3, x_dev_3,label_dev_3, 100, 'syllable_cnn_train_3.model')
#model_4=cnn_char_seg.cnn_train(x_train_4,label_train_4, x_dev_4,label_dev_4, 100, 'syllable_cnn_train_4.model')
#model_5=cnn_char_seg.cnn_train(x_train_5,label_train_5, x_dev_5,label_dev_5, 100, 'syllable_cnn_train_5.model')
#test=cnn_char_seg.cnn_test(x_test_1,label_test_1,1,'syllable_cnn_train_5.model',cnn_char_seg)#
#new_dict=cnn_char_seg.new_dict('decode_token_1.lst','new_syllable_5.dict','new_id_syllable_5.dict','id_word_pos_combine.dict')
#decode=cnn_char_seg.decode_syllable(x_test_5,label_test_5,1,'syllable_cnn_train_5.model',cnn_char_seg,'decode_syllable_5.lst')
#decode_file=cnn_char_seg.decode_data('decode_syllable_1.txt', 'save_decode_syllable_1.txt','lao_syllable_unk.dict','new_lao_sylable_dict_1.dict','_new_id_sylable_dict_1.dict','ws_cnn_train_w_unk_1.model',cnn_char_seg,label_test_1,x_test_1)
#decode_word=cnn_char_seg.decode_word('unk_lao_syllable.dict','save_decode_syllable_1.txt')
#print (x_test_1)


