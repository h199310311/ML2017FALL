import csv
import numpy as np
from sys import argv

def sigmoid(z):
    sig = 1/(1+np.exp((-1)*z))
    return np.clip(sig,0.00000000000001,0.99999999999999)

# training data
reader = csv.reader(open(argv[3], mode='r',))
next(reader)
tra_in = []
for row in reader:
    new = []
    i=0
    for elem in row:
        if (i!=14)and(i!=52)and(i<64) :
            new.append(float(elem))
        i+=1
    tra_in.append(new)
tra_in = np.array(tra_in)

reader = csv.reader(open(argv[4], mode='r',))
next(reader)
tra_ou = []
for row in reader:
    for elem in row:
        tra_ou.append(float(elem))
tra_ou = np.array(tra_ou)

#normalize
normalize = np.zeros((2,6), dtype='f')
for i in range(len(normalize[0])):
    min = tra_in[0][i]
    max = tra_in[0][i]
    for j in range(len(tra_in)):
        if tra_in[j][i] < min:
            min = tra_in[j][i]
        if tra_in[j][i] > max:
            max = tra_in[j][i]
    normalize[0][i] = min
    normalize[1][i] = max
    cover = max - min
    for j in range(len(tra_in)):
        tra_in[j][i] = (tra_in[j][i]-min)/cover

# add bias
tra_in = np.concatenate((np.ones((tra_in.shape[0],1)),tra_in), axis=1)

# gradient
w = [0.1 for n in range(len(tra_in[0]))]
w = np.array(w)
lr = 1
iteration = 10000
balance=0
tra_in_t = tra_in.transpose()
s_gra = np.zeros(len(tra_in[0]))

for i in range(iteration):
    hypo = np.dot(tra_in,w)
    sig = sigmoid(hypo)
    gra = np.dot(tra_in_t,((-1)*(tra_ou-sig))) + balance*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - lr * gra/ada

# save Model
np.save('normalize.npy',normalize)
np.save('model.npy',w)