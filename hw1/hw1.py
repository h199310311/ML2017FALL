import csv
import numpy as np
from sys import argv

get_time = 9
feature = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]

w = np.load('model.npy')

test_x = []
n_row = 0
text = open(argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

# 整理testing data
x = 0
for i in row:
    if (x%18) == 0:
        test_x.append([])
    for k in range(11-get_time,11):
        if feature[x%18] == 1:
            if i[k] !="NR":
                test_x[int(x/18)].append(float(i[k]))
            else:
                test_x[int(x/18)].append(0)
    x = x+1

test_x = np.array(test_x)
                   
# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

#輸出結果
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(test_x[i],w)
    ans[i].append(a)

text = open(argv[2], "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()