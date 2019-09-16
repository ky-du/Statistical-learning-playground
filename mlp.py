# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt

#implement of MLP(original form)
#the goal of MLP is to find w* and b*

#initialization
w=[1,1]
b=0
learning_rate=1
max_epoch=100

#generate linear separable data with fixed random seed
def data_generator():
    random.seed()
    generated_data=datasets.make_classification(n_samples=100, n_features=2, 
                                                n_informative=2, n_redundant=0, 
                                                n_classes=2, random_state=0,
                                                flip_y=0, class_sep=1.5,
                                                n_clusters_per_class=1) 
    data,label = generated_data
    
    #plot
    color = ['red','blue']
    for i in range(2):
        plt.scatter(data[:,0][label == i], data[:,1][label == i], c=color[i])
    plt.title('generated data')
    label[label==0]=-1
    
    plt.show()
    return data,label


#loss function defined in page 37
def loss(w,b,x,y):
    error_list=np.zeros(len(x))
    loss_value=0
    for i in range(len(x)):
        if not(np.sign(np.dot(w,x[i])+b)==y[i]):
            error_list[i]=1
            loss_value+=-y[i]*(np.dot(w,x[i])+b)
    return error_list, loss_value
        

#show decision_boundry and its classification result, datapoint marked with stars
#are mis-classified samples.
def show_classification(w,b,data,label,error_list,epoch):
    
    label_value=[-1,1,-1,1]
    error_flag=[0,0,1,1]
    color_and_marker=[['red','o'], ['blue','o'], ['red','*'], ['blue','*']]
    
    # 2x2 conditions
    for i in range(4):
        plt.scatter(data[:,0][np.all([(label == label_value[i]),(error_list==error_flag[i])],axis=0)], 
                    data[:,1][np.all([(label == label_value[i]),(error_list==error_flag[i])],axis=0)], 
                    c=color_and_marker[i][0], marker=color_and_marker[i][1])
    
    decision_boundry_x=np.linspace(data[:,0].min(),data[:,0].max())
    decision_boundry_y= -(w[0]*decision_boundry_x+b)/w[1]
    plt.plot(decision_boundry_x, decision_boundry_y)
    plt.title('classification result of epoch ' +str(epoch))
    plt.show()
    
    
if __name__=="__main__":
    data,label=data_generator()
    error_list, loss_value = loss(w,b,data,label)
    show_classification(w,b,data,label,error_list,0)    
    num_epoch=0
    
    #training 
    loss_data=[]
    for i in range(max_epoch):
        
        # parameter update
        w += learning_rate*np.sum(data[error_list==1]* label[error_list==1][:, None],axis=0)
        b += learning_rate*np.sum(label[error_list==1])
        
        error_list, loss_value = loss(w,b,data,label)
        show_classification(w,b,data,label,error_list,i+1)
        
        loss_data.append(loss_value)
        
        if loss_value==0:
            break
        
    plt.plot(loss_data)
    plt.title('empirical loss')
    plt.show()
        
