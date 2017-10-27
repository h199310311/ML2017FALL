import csv
import numpy as np
from sys import argv

# testing data
w = np.load('model.npy')
normalize = np.load('normalize.npy')

reader = csv.reader(open(argv[1], mode='r',))
next(reader)
test_x = []
for row in reader:
    new = []
    i = 0
    for elem in row:
        if (i!=14)and(i!=52)and(i<64) :
            new.append(float(elem))
        i+=1
    test_x.append(new)
test_x = np.array(test_x)

#normalize
for i in range(len(normalize[0])):
    for j in range(len(test_x)):
        test_x[j][i] = (test_x[j][i]-normalize[0][i])/(normalize[1][i]-normalize[0][i])

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

#判斷結果
ans = np.dot(test_x,w)
sig = 1/(1+np.exp((-1)*ans))
dis_ans = np.ones(len(sig), dtype='i')
for i in range(len(sig)):
    if sig[i] < 0.5:
        dis_ans[i] = 0

#輸出結果
fin_ans = []
for i in range(len(test_x)):
    fin_ans.append([str(i+1)])
    fin_ans[i].append(dis_ans[i])

text = open(argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(fin_ans)):
    s.writerow(fin_ans[i])
text.close()