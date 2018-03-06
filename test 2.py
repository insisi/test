#-*-coding: utf-8 -*-
import codecs
import re
import sys
import optparse
import os
import  tempfile
import string
import subprocess
from collections import defaultdict
combine_sent = []
token_sent=[]
token_dict=defaultdict(list)
#sent=[['a','B-N'],['b','I-N'],['c','I-N'],['d','B-ADJ'],['e','I-ADJ']]
predict=[(0,1),(9,1),(0,3),(2,1)]
target=[(3,1),(4,1),(1,3),(5,1), (2,1)]
total_token=set(predict+target)
for i,token in enumerate(total_token):
	token_dict[token]=i
token_dict=dict(token_dict)
print (token_dict)

# for i in predict:
# 	token_sent.append(i)
# for i in target:
# 	token_sent.append(i)
# print (token_sent)
# sent=[[['a','B-N'],['b','I-N'],['c','I-N'],['d','B-ADJ'],['e','I-ADJ']],[['y','B-V'],['z','I-PRE'],['j','B-V'],['k','I-V']]]
# for s in sent:
# 	w = []
# 	ws = []
# 	for c,n in s:
# 		if n[0]=='B':
# 			if w:
# 				ws.append(w)
# 			w=[]
# 		w.append(n[2:])
# 	ws.append(w)
# 	combine_sent.append(ws)
# for i in range(len(combine_sent)):
# 	for j in range(len(combine_sent[i])):
# 		if (len(set(combine_sent[i][j])))==1:
# 			combine_sent[i][j]=combine_sent[i][j][0]
# 		else:
# 			combine_sent[i][j]="NULL"
# print (combine_sent)

# for s in sent:
#     w = []
#     ws = []
#     for c, n in s:
#         if n[0] == 'B':
#             if w:
#                 ws.append("".join(w))
#             w = []
#         w.append(c)
#     ws.append("".join(w))
#     token_sent.append(ws)
# print (token_sent)
	# for j in i:
	# 	print (len(set(j)))
# print (combine_sent)
	# 	if n=='B':
	# 		if w:
	# 	 		ws.append("".join(w))
	# 	 	w=[]
	# 	w.append(c)
	# ws.append("".join(w))
	# combine_sent.append(ws)



# 	w=[]
# 	ws=[]
#     for c, n in s:
 
 #        if n == "B":
 #            if w:
 #                ws.append("".join(w))
 #            w = []
 #        w.append(c)
 #    ws.append("".join(w))
 #    combine_sent.append(ws)