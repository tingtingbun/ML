
import numpy as np
import sys

if __name__ == '__main__':


    with open (sys.argv[1],"rt") as input1:
        list1 = []
        for line in input1:
            line = line[:-1].split(',')
            label=np.zeros((10,1))
            label[int(line[0])]=1
            att=[]
            for i in line[1:]:
                att.append(float(i))
            total=[label,np.asarray(att)]
            list1.append(total)
        
        train=list1





    with open (sys.argv[2],"rt") as input2:
        list1 = []
        for line in input2:
            line = line[:-1].split(',')
            label=np.zeros((10,1))
            label[int(line[0])]=1
            att=[]
            for i in line[1:]:
                att.append(float(i))
            total=[label,np.asarray(att)]
            list1.append(total)
        
        test=list1


# In[36]:


    def sigmoid_forward(x):
        return 1.0/(1+ np.exp(-x))

    def linear_forward(x,para):
        return np.dot(para,x)

    def softmax_forward(x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def cross_entro_forward(ds,al,be):
        trainj=0.0
        for i in ds:
            x=i[1].copy()
            x=np.append(x,1).reshape((x.shape[0]+1,1))
            y=i[0].copy()
            a,z,z1,b,pred=forward(x,y,al,be)
            trainj+=float(-np.log(pred[np.argmax(y),]))        
        return trainj/len(ds)

    def sigmoid_backward(sig):
        return sig * (1.0 - sig)


# In[37]:


    def forward(x,y,alpha,beta):
        a=linear_forward(x,alpha)
        z=sigmoid_forward(a)
        z1=np.append(z,1).reshape((z.shape[0]+1,1))
        b=linear_forward(z1,beta)
        pred=softmax_forward(b)
      
        return a,z,z1,b,pred


# In[38]:


    def backward(y,pred,z,z1,x,beta):
        d_beta = np.dot((pred-y), z1.T) 
        d_alpha = np.dot((np.dot(beta[:,:-1].T,(pred-y))) * sigmoid_backward(z), x.T)
    
        return d_beta, d_alpha


# In[39]:


    def Initialization(flag,hidden_unit):
        if flag == 1:
            alpha = np.random.uniform(-0.1,0.1,size=(hidden_unit,(128)))
            alpha=np.append(alpha,np.zeros([len(alpha),1]),1)
            beta=np.random.uniform(-0.1,0.1,size=(10,hidden_unit))
            beta=np.append(beta,np.zeros([len(beta),1]),1)
    
        elif flag == 2:
            alpha = np.zeros((hidden_unit,(128+1)))
            beta=np.zeros((10,hidden_unit+1))
    
        return alpha,beta


# In[93]:


    def Error(train_data,test_data,alpha,beta):
        n=0
        m=0
        for ex in train_data:       
                x=ex[1].copy()
                x=np.append(x,1).reshape((x.shape[0]+1,1))
                y=ex[0].copy()
                a,z,z1,b,pred=forward(x,y,alpha,beta)
                if np.argmax(pred) != np.argmax(y):
                    n+=1
        for ex in test_data:       
                x=ex[1].copy()
                x=np.append(x,1).reshape((x.shape[0]+1,1))
                y=ex[0].copy()
                a,z,z1,b,pred=forward(x,y,alpha,beta)
                if np.argmax(pred) != np.argmax(y):
                    m+=1
        return [(n/len(train_data)),(m/len(test_data))]
            


# In[106]:


    def predict(ds,al,be):
        pr=[]
        for i in ds:
            x=i[1].copy()
            x=np.append(x,1).reshape((x.shape[0]+1,1))
            y=i[0].copy()
            a,z,z1,b,pred=forward(x,y,al,be)
            pr.append(np.argmax(pred))
        return pr


# In[88]:


    def SGD(train_data,test_data,flag,hidden_unit,epoch,rate):
        al,be=Initialization(flag,hidden_unit)
        num=0
        entropy_train=[]
        entropy_test=[]
        while num < epoch:
            num +=1
            list1=[]
            for ex in train_data:       
                x=ex[1].copy()
                x=np.append(x,1).reshape((x.shape[0]+1,1))
                y=ex[0].copy()
                a,z,z1,b,pred=forward(x,y,al,be)
                list1.append(np.argmax(pred))
                d_beta,d_alpha=backward(y,pred,z,z1,x,be)
                al-=rate*d_alpha
                be-=rate*d_beta
            entropy_train.append(cross_entro_forward(train_data,al,be))
            entropy_test.append(cross_entro_forward(test_data,al,be))
        return al,be,entropy_train,entropy_test
        


# In[98]:


    a1,b1,en_tr,en_te=SGD(train,test,int(sys.argv[8]),int(sys.argv[7]),int(sys.argv[6]),float(sys.argv[9]))


# In[99]:


    err=Error(train,test,a1,b1)


# In[101]:


    with open (sys.argv[5],"wt") as output1:
        for num in range(len(en_tr)):
            output1.write("epoch="+str(int(num)+1)+" "+"crossentropy(train):"+" "+str(en_tr[int(num)])+'\n')
            output1.write("epoch="+str(int(num)+1)+" "+"crossentropy(test):"+" "+str(en_te[int(num)])+'\n')
        output1.write("error(train):"+" "+str(err[0])+'\n')
        output1.write("error(test):"+" "+str(err[1]))


# In[108]:


    with open (sys.argv[3],"wt") as output2:
        p=predict(train,a1,b1)
        for i in p:
            output2.write(str(i)+'\n')
    


# In[109]:


    with open (sys.argv[4],"wt") as output3:
        for i in predict(test,a1,b1):
            output3.write(str(i)+'\n')
    

