import csv
import numpy as np
from sys import argv
from scipy.stats import norm

des_in = np.load('Generative descrete model.npy')
con_in = np.load('Generative continuous model.npy')

# testing data
reader = csv.reader(open(argv[1], mode='r',))
next(reader)
test_in_des = []
test_in_con = []
for row in reader:
    new_des = []
    new_con = []
    i = 0
    for elem in row:
        if i==0 or i==1 or i==3 or i==4 or i==5:
            new_con.append(float(elem))
        else:
            new_des.append(float(elem))
        i+=1
    test_in_des.append(new_des)
    test_in_con.append(new_con)
test_in_des = np.array(test_in_des)
test_in_con = np.array(test_in_con)

#計算機率
res=[]
for i in range(len(test_in_des)):
    c1_total = 1 
    c2_total = 1 
    for j in range(len(test_in_con[0])):
        c1 = (norm.pdf(test_in_con[i][j],con_in[0][j+1],con_in[0][j+6]))*con_in[0][0]
        c2 = (norm.pdf(test_in_con[i][j],con_in[1][j+1],con_in[1][j+6]))*con_in[1][0]
        c1_total *= (c1/(c1+c2))
        c2_total *= (c2/(c1+c2))
    for j in range(len(test_in_des[0])):
        if test_in_des[i][j]==1:
            c1_total *=des_in[0][j]
            c2_total *=des_in[2][j]
    if c1_total>c2_total:
        res.append(1)
    else:
        res.append(0)

#輸出結果
fin_ans = []
for i in range(len(res)):
    fin_ans.append([str(i+1)])
    fin_ans[i].append(res[i])

text = open(argv[2], "w+")
s = csv.writer(open(argv[2], "w+"),delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(fin_ans)):
    s.writerow(fin_ans[i])
text.close()