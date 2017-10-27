import csv
import numpy as np
from sys import argv

# training data
reader = csv.reader(open(argv[1], mode='r',))
next(reader)
tra_in_des = []
tra_in_con = []
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
    tra_in_des.append(new_des)
    tra_in_con.append(new_con)
tra_in_des = np.array(tra_in_des)
tra_in_con = np.array(tra_in_con)

reader = csv.reader(open(argv[2], mode='r',))
next(reader)
tra_ou = []
for row in reader:
    for elem in row:
        tra_ou.append(float(elem))
tra_ou = np.array(tra_ou)

#計算連續型資料
class1 = 0
class2 = 0
c1_mean = np.zeros(5)
c2_mean = np.zeros(5)
c1_var = np.zeros(5)
c2_var = np.zeros(5)
for i in range(len(tra_ou)):
    if tra_ou[i]==1 :
        class1+=1
        c1_mean+=tra_in_con[i]
        c1_var+=tra_in_con[i]**2
    else :
        class2+=1
        c2_mean+=tra_in_con[i]
        c2_var+=tra_in_con[i]**2
for i in range(5):
    c1_mean[i]/=class1
    c2_mean[i]/=class2
    c1_var[i]/=class1
    c2_var[i]/=class2
c1_var =  (c1_var-(c1_mean**2))**(0.5)
c2_var =  (c2_var-(c2_mean**2))**(0.5)

pro=np.array([[float(class1)/len(tra_ou)],[float(class2)/len(tra_ou)]])
mean=np.row_stack((c1_mean,c2_mean))
var=np.row_stack((c1_var,c2_var))
con_ou=np.column_stack((pro,mean,var))

#計算離散型資料
des_ou=[[],[],[],[]]
for i in range(len(tra_in_des[0])):
    count_c1x1 = 0
    count_c1x0 = 0
    count_c2x1 = 0
    count_c2x0 = 0
    for j in range(len(tra_in_des)):
        if tra_in_des[j][i]==1:
            if tra_ou[j]==1:
                count_c1x1+=1
            else:
                count_c2x1+=1
        else:
            if tra_ou[j]==1:
                count_c1x0+=1
            else:
                count_c2x0+=1
    des_ou[0].append(float(count_c1x1)/(count_c1x1+count_c2x1))
    des_ou[1].append(float(count_c1x0)/(count_c1x0+count_c2x0))
    des_ou[2].append(float(count_c2x1)/(count_c1x1+count_c2x1))
    des_ou[3].append(float(count_c2x0)/(count_c1x0+count_c2x0))
des_ou=np.array(des_ou)

# save Model
np.save('Generative descrete model.npy',des_ou)
np.save('Generative continuous model.npy',con_ou)