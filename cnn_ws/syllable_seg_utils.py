#-*-coding: utf-8 -*-
import codecs
import re
import sys
import optparse
import os
import  tempfile
import string
import pickle

alphabet=list(string.ascii_lowercase)+list(string.ascii_uppercase)
ld_list=['໐','໑','໒','໓','໔','໕','໖','໗','໘','໙']
q_list=list(string.punctuation)+['“','’','”','…']
d_list=['1','2','3','4','5','6','7','8','9','0']
marks_lst=list(string.punctuation)+['“','’','”']
w_list=['_']

def replace_num_alphabet(corpus_file,output_file,col):
    sys.stdout = codecs.open(output_file, 'w', encoding='utf-8')
    sent_lst = [[]]
    num = 0
    syllable_lst = []
    for line in codecs.open(corpus_file, 'r', encoding='utf-8'):
        line = line.strip().split()
        if len(line) != col:
            num += 1
            sent_lst.append([])
            continue
        sent_lst[num].append([line[3], line[4],line[2]])
    for i in range(len(sent_lst)):
        for j in range(len(sent_lst[i])):
            if sent_lst[i][j][0][0] in q_list+w_list:
                sent_lst[i][j][0] = 'symbol'
            # elif sent_lst[i][j][0].isdigit():
            #     sent_lst[i][j][0]='number'
            # elif sent_lst[i][j][0][0] in alphabet:
            #     sent_lst[i][j][0]='alphabet'
            # elif sent_lst[i][j][0][0].isdigit():
            #     sent_lst[i][j][0]='number'
    for i in range(len(sent_lst)):
        for j in range(len(sent_lst[i])):
            print (j+1,sent_lst[i][j][0],sent_lst[i][j][2],sent_lst[i][j][1])
        print (' ')
def syllable_bi(corpus_file,output_file):
    sys.stdout = codecs.open(output_file, 'w', encoding='utf-8')
    sent_lst=[[]]
    num=0
    syllable_lst=[]
    for line in codecs.open(corpus_file,'r',encoding='utf-8'):
        line=line.strip().split()
        if len(line)!=4:
            num+=1
            sent_lst.append([])
            continue
        sent_lst[num].append([line[1],line[2]])
    for i in range(len(sent_lst)):
        for j in range(len(sent_lst[i])):
            if sent_lst[i][j][1]=='0':
                print (sent_lst[i][j][0]+'\t'+'0')
            else:
                print(sent_lst[i][j][0] + '\t' + '1')
        print (' ')

def char_syllable_bi(corpus_file,output_file):
    sys.stdout = codecs.open(output_file, 'w', encoding='utf-8')
    sent_lst=[[]]
    num=0
    syllable_lst=[]
    for line in codecs.open(corpus_file,'r',encoding='utf-8'):
        line=line.strip().split()
        if len(line)!=3:
            num+=1
            sent_lst.append([])
            continue
        sent_lst[num].append([line[0],line[1]])
    for i in range(len(sent_lst)):
        for j in range(len(sent_lst[i])):
            if sent_lst[i][j][1] not in ['number','alphabet','symbol']:
                for k in range(len(sent_lst[i][j][1])):
                    if k==0:
                        print (sent_lst[i][j][1][k]+'\t'+'0')
                    else:
                        print(sent_lst[i][j][1][k] + '\t' + '1')
            else:
                print (sent_lst[i][j][1]+'\t'+'0')
        print (' ')

def char_id_dict(syllable_bi_file,syllable_id,id_syllable):
    char_list=[]
    char_id={}
    id_char={}
    for i in codecs.open(syllable_bi_file,'r',encoding='utf-8'):
        i=i.strip().split("\t")
        if len(i)>1:
            char_list.append(i[0])
    char_list=set(char_list)
    char_list=list(char_list)
    total=len(char_list)
    print (total)
    char_id['</s>']=0
    for i,char in enumerate(char_list):
        char_id[char]=i+1
    # char_id['</s>']=total
    pickle.dump(char_id,open(syllable_id,"wb"))
    for char, id in char_id.items():
        id_char[id] = char
    pickle.dump(id_char, open(id_syllable, "wb"))
    #print (char_id)
    return char_id,id_char
def label_num(char_bi_file,label_emb):
    num=0
    label_list=[[]]
    for i in codecs.open(char_bi_file,'r',encoding='utf-8'):
        i=i.strip().split("\t")
        if len(i)!=2:
            num+=1
            label_list.append([])
            continue
        label_list[num].append(int(i[1]))
    pickle.dump(label_list, open(label_emb, "wb"))
    return label_list
def char_n_padding(char_bi_file,num_padding,output_no_padding,output_with_padding):
    num=0
    char=[[]]
    label=[[]]
    for i in codecs.open(char_bi_file,'r',encoding='utf-8'):
        i=i.strip().split("\t")
        if len(i)!=2:
            num+=1
            char.append([])
            label.append([])
            continue
        char[num].append(i[0]) #append char
        label[num].append(i[1]) #append label
    pickle.dump(char,open(output_no_padding,"wb")) #with no padding
    ##pickle.dump(label,open("label.lst","wb")) #with no padding
    for m in range(len(char)):
        for k in range(num_padding):
            char[m].insert(0,'</s>')# add n padding before char
            char[m].append('</s>') #add n padding after char
    pickle.dump(char,open(output_with_padding,"wb"))
    return char
#convert to a sentences that has n padding
def sent_id(char_id_dict,char_2_padding,sentence_id):
    count=0
    char_list=pickle.load(open(char_2_padding,'rb'))#sent lists that have 2 padding
    char_dict=pickle.load(open(char_id_dict,'rb'))#character dictionary
    for i in range(len(char_list)):
        for j in range(len(char_list[i])):
            if  char_list[i][j] in char_dict:
                char_list[i][j]=char_dict[char_list[i][j]]
    pickle.dump(char_list,open(sentence_id,"wb"))
    return char_list

def get_pos_feature(corpus_file,save_output):
    sys.stdout = codecs.open(save_output, 'w', encoding='utf-8')
    pronoun_lst=['ຂ້ອຍ','ເຈົ້າ','ເຮົາ','ມັນ','ໂຕ','ກູ','ມຶ່ງ','ສູ','ຂ້າພະເຈົ້າ','ຂ້ານ້ອຍ','ອ້າຍ','ເອື້ອຍ','ນ້ອງ']
    prossessive_lst=['ຂອງຂ້ອຍ','ຂອງເຈົ້າ','ຂອງເຮົາ','ຂອງມັນ','ຂອງໂຕ','ຂອງກູ','ຂອງມຶ່ງ','ຂອງສູ','ຂອງຂ້າພະເຈົ້າ','ຂອງຂ້ານ້ອຍ','ຂອງອ້າຍ','ຂອງເອື້ອຍ','ຂອງນ້ອງ','ຂອງລາວ']
    neg_lst=['ບໍ່','ບໍ່ໄດ້']
    title_lst=['ທ້າວ', 'ນາງ', 'ບັກ', 'ອີ່','ປ້າ','ລຸງ','ແມ່','ພໍ່', 'ພໍ່ຕູ້','ນາຍໝໍ','ສາດສະຕາຈານ']
    sent=[[]]
    num=0
    for line in codecs.open(corpus_file, 'r', encoding='utf-8'):
        line=line.strip().split(" ")
        if len(line)!=3:
            num+=1
            sent.append([])
            continue
        sent[num].append([line[1],line[0],line[2]])
    for i in range(len(sent)):
        for j in range(len(sent[i])):
            if len(sent[i][j][0])>4 and sent[i][j][0][:4]=='ຄວາມ':
                sent[i][j][1]='noun'
            elif len(sent[i][j][0])>4 and sent[i][j][0][:4]=='ຢ່າງ':
                sent[i][j][1] = 'adverb'
            elif len(sent[i][j][0])>4 and sent[i][j][0][:3]=='ການ':
                sent[i][j][1] = 'noun'
            elif sent[i][j][0] in prossessive_lst:
                sent[i][j][1] = 'pronoun'
            elif sent[i][j][0] in pronoun_lst:
                sent[i][j][1]='pronoun'
            elif sent[i][j][0].isdigit():
                sent[i][j][1] = 'number' #cnm is digit
            elif sent[i][j][0][0] in d_list:
                sent[i][j][1] = 'number'
            elif sent[i][j][0] in neg_lst:
                sent[i][j][1] = 'negative'
            elif sent[i][j][0] in title_lst:
                sent[i][j][1] = 'title'
            elif sent[i][j][0] in q_list+marks_lst+w_list:
                sent[i][j][1] = 'symbol'
            else:
                sent[i][j][1]='-'
            print ("%s %s %s"%(sent[i][j][0],sent[i][j][1],sent[i][j][2]))
        print (" ")
def decode_data(load_file,save_file):
    sys.stdout = codecs.open(save_file, 'w',encoding='utf-8')
    char_c=[]
    list_c=[]
    b=[[]]
    num=0
    for line in codecs.open(load_file,'r',encoding='utf-8'):
        line=line.strip().split()
        if len(line)==3:
            list_c.append([line[0],line[1]])
    for i in range(len(list_c)):
        if list_c[i][1]=="0":
            char_c.append(list_c[i][0])
            num+=1
            b.append([])
            continue
        b[num].append(list_c[i][0])
    b.pop(0)
    for k in range(len(b)):
        b[k].insert(0,char_c[k])
    #print b

    for i in range(len(b)):
        for j in range(len(b[i])):
            sys.stdout.write(b[i][j])
        print (' ')

def get_pos_feature_raw(corpus_file,save_output):
    sys.stdout = codecs.open(save_output, 'w', encoding='utf-8')
    pronoun_lst=['ຂ້ອຍ','ເຈົ້າ','ເຮົາ','ມັນ','ໂຕ','ກູ','ມຶ່ງ','ສູ','ຂ້າພະເຈົ້າ','ຂ້ານ້ອຍ','ອ້າຍ','ເອື້ອຍ','ນ້ອງ']
    prossessive_lst=['ຂອງຂ້ອຍ','ຂອງເຈົ້າ','ຂອງເຮົາ','ຂອງມັນ','ຂອງໂຕ','ຂອງກູ','ຂອງມຶ່ງ','ຂອງສູ','ຂອງຂ້າພະເຈົ້າ','ຂອງຂ້ານ້ອຍ','ຂອງອ້າຍ','ຂອງເອື້ອຍ','ຂອງນ້ອງ','ຂອງລາວ']
    neg_lst=['ບໍ່','ບໍ່ໄດ້']
    title_lst=['ທ້າວ', 'ນາງ', 'ບັກ', 'ອີ່','ປ້າ','ລຸງ','ແມ່','ພໍ່', 'ພໍ່ຕູ້','ນາຍໝໍ','ສາດສະຕາຈານ']
    sent=[]
    num=0
    for line in codecs.open(corpus_file, 'r', encoding='utf-8'):
        line=line.strip()
        sent.append(line)
    for i in range(len(sent)):
        if len(sent[i])>4 and sent[i][:4]=='ຄວາມ':
            print ("%s %s %s"%(sent[i],'noun',sent[i]))
        elif len(sent[i])>4 and sent[i][:4]=='ຢ່າງ':
            print ("%s %s %s" % (sent[i], 'adverb', sent[i]))
        elif len(sent[i])>4 and sent[i][:3]=='ການ':
            print ("%s %s %s" % (sent[i], 'noun', sent[i]))
        elif sent[i] in prossessive_lst:
            print ("%s %s %s" % (sent[i], 'pronoun', sent[i]))
        elif sent[i] in pronoun_lst:
            print ("%s %s %s" % (sent[i], 'pronoun', sent[i]))
        elif sent[i].isdigit():
            print ("%s %s %s" % (sent[i], 'number', sent[i]))
        elif sent[i][0] in d_list:
            print ("%s %s %s" % (sent[i], 'number', sent[i]))
        elif sent[i] in neg_lst:
            print ("%s %s %s" % (sent[i], 'negative', sent[i]))
        elif sent[i] in title_lst:
            print ("%s %s %s" % (sent[i], 'title', sent[i]))
        elif sent[i] in q_list+marks_lst+w_list:
            print ("%s %s %s" % (sent[i], 'symbol', sent[i]))
        else:
            print ("%s %s %s" % (sent[i], '-', sent[i]))

def evaluate_seg_pos(decode_file, test_file):
    #sys.stdout = codecs.open('get_test.txt', 'w', encoding='utf-8')
    decode_sent=[]
    test_sent=[]
    test_token=[]
    decode_token=[]
    num=0
    num_1=0
    for line in codecs.open(decode_file, 'r', encoding='utf-8'):
        line=line.strip().split()
        if len(line)==4:
            decode_sent.append([line[0],line[3]])
            decode_token.append(line[0])

    for line in codecs.open(test_file, 'r', encoding='utf-8'):
        line = line.strip().split()
        if len(line) == 3:
            test_sent.append([line[1],line[2]])
            test_token.append(line[1])
    #print (test_sent)
    for i in range(len(decode_sent)):
        #print (decode_sent[i])
        if decode_sent[i] in test_sent:
            num+=1
    print (num)
    print (decode_sent)
    print (test_sent)
            #
    # for i in range(len(decode_sent)):
    # 	if decode_sent[i][0] in test_sent:
    # 		if decode_sent[i][1]==test_sent[decode_sent[i][0]]:
    # 			num += 1
    total=len(test_sent)
    sys_out=len(decode_sent)
    recall=float(num)/total
    precision=float(num)/sys_out
    print ('system',sys_out)
    print ('total',total)
    print ('recall',recall)
    print ('precision', precision)
    print (len(test_token))
def convert_bi(input_file,output_file):
    sys.stdout = codecs.open(output_file, 'w', encoding='utf-8')
    sent = [[]]
    num = 0
    for line in codecs.open(input_file, 'r', encoding='utf-8'):
        line = line.strip().split()
        if len(line) != 2:
            num += 1
            sent.append([])
            continue
        sent[num].append([line[0], line[1]])
    for i in range(len(sent)):
        for j in range(len(sent[i])):
            if sent[i][j][1]=='B':
                print (sent[i][j][0]+'\t'+'0')
            else:
                print (sent[i][j][0]+'\t'+'1')
        print (' ')

if __name__ == '__main__':
    # corpus_file='final_lao_corpus_3k.txt'
    # output_file='final_lao_corpus_3k_number_replaced.txt'
    # syllable_output_file='syllable_corpus.txt'
    syllable_bi_file='char_01.txt'
    syllable_dict='char_syllable.dict'
    id_syllable='id_char_syllable.dict'
    label_emb='char_syllable_lable_emb.lst'
    output_no_padding='char_syllable_no_pad.lst'
    output_with_padding='char_syllable_2_padding.lst'
    sentence_id_2_padding='char_sent_id_2_padding.lst'
    #convert_bi('char_bi.txt', 'char_01.txt')
    #char_syllable_bi('syllable_corpus.txt', 'char_syllable_bi.txt')
    #replace_num_alphabet('final_check_syllable_fixed.txt', 'syllable_corpus.txt',5)
    #syllable_bi(syllable_output_file,syllable_bi_file)
    char_id_dict(syllable_bi_file, syllable_dict,id_syllable)
    label_num(syllable_bi_file, label_emb)
    char_n_padding(syllable_bi_file, 2, output_no_padding, output_with_padding)
    sent_id(syllable_dict, output_with_padding, sentence_id_2_padding)

    #get_pos_feature('crf_tarin_replace_5.txt', 'crf_tarin_feat_5.txt')
    #decode_data('cnn_syllable_seg_2.txt', 'decode_cnn_syllable_seg_2.txt')
    #get_pos_feature_raw('decode_cnn_syllable_seg_5.txt', 'decode_cnn_syllable_feat_5.txt')
    #evaluate_seg_pos('output_cnn_syllable_wp_5.txt', 'crf_test_replace_5.txt')
    # x_input = pickle.load(open('char_syllable_lable_emb.lst', 'rb'))
    # print (x_input)