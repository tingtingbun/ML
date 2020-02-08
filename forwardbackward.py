import numpy as np
import sys

if __name__ == '__main__':
    with open (sys.argv[3],"rt") as tagset, open (sys.argv[2],"rt") as wordset:
        tag_list=[]
        word_list=[]
        for tag_line in tagset:
        #if index
            if tag_line[-1]=='\n':
                temp_tag=tag_line[:-1]
            else:
                temp_tag=tag_line
            tag_list.append(temp_tag)
        for word_line in wordset:
            if word_line[-1]=='\n':
                temp_word=word_line[:-1]
            else:
                temp_word=word_line
            word_list.append(temp_word)

    with open (sys.argv[1],"rt") as data:
        dataset=[]
        for line in data:
            if line[-1]=='\n':
                temp = line[:-1].split(' ')
            else:
                temp = line.split(' ')
       
            temp_word=[]
            temp_tag = []
        
            for ele in temp:
                word=ele.split('_')[0]
                tag=ele.split('_')[1]
           
                temp_tag.append(tag_list.index(tag))
                temp_word.append(word_list.index(word))
                seq=list(zip(temp_word,temp_tag))
            
            dataset.append(seq)

    
    pi_test=np.loadtxt(sys.argv[4],delimiter=' ')

    
    b_test=np.loadtxt(sys.argv[5],delimiter=' ')
    
    a_test=np.loadtxt(sys.argv[6],delimiter=' ')

    def forward(index,al,a,b,seq,end_index):
    
        if index == end_index+1:
            return (al)
        else:
            alpha_new = b[:,seq[index][0]].reshape(len(b),1)*np.dot(a.T,al)
            return(forward(index+1,alpha_new,a,b,seq,end_index))
        

    def backward(beg_index,be,a,b,seq,end_index):
        if beg_index == end_index-1:
            return (be)
        else:
            beta_new = np.dot(a,(b[:,seq[beg_index+1][0]].reshape(len(b),1)*be))
            
            return(backward(beg_index-1,beta_new,a,b,seq,end_index))
        

    def predict(test_ds,a,b,pi,word_ds,tag_ds):
        out_list=[]
        for i in test_ds:
            alpha_init=b[:,i[0][0]].reshape(len(b),1)*pi.reshape(len(pi),1)
            beta_init=np.ones((len(b),1))
            inst_list=[]
            for ind,val in enumerate(i):
                fo_arr=forward(1,alpha_init,a,b,i,ind)
                ba_arr=backward(len(i)-2,beta_init,a,b,i,ind)
                total_pro = fo_arr*ba_arr
                output=[word_ds[val[0]],tag_ds[np.argmax(total_pro)]]
                output='_'.join(output)
                inst_list.append(output)
            out_list.append(inst_list)
        return out_list

    def error_rate(ds,tag_ds,a,b,pi):
        error=0
        total=0
        for i in ds:
            alpha_init=b[:,i[0][0]].reshape(len(b),1)*pi.reshape(len(pi),1)
            beta_init=np.ones((len(b),1))
        
            for ind,val in enumerate(i):
                total+=1
                fo_arr=forward(1,alpha_init,a,b,i,ind)
                ba_arr=backward(len(i)-2,beta_init,a,b,i,ind)
                total_pro = fo_arr*ba_arr
            
                if tag_ds[val[1]]!=(tag_ds[np.argmax(total_pro)]):
                    error+=1
        
        return 1-error/total

    def log_likelihood (ds,a,b,pi):
        ds_pro=[]
        for i in ds:
            alpha_init=b[:,i[0][0]].reshape(len(b),1)*pi.reshape(len(pi),1)
            beta_init=np.ones((len(b),1))
            seq_val=0
            for ind,val in enumerate(i):
                if ind == len(i)-1:
                    fo_arr=forward(1,alpha_init,a,b,i,ind)
                    seq_val=(np.log(np.sum(fo_arr)))
            ds_pro.append(seq_val)
        return (sum(ds_pro)/len(ds_pro))
        

    a=predict(dataset,a_test,b_test,pi_test,word_list,tag_list)
    log=log_likelihood(dataset,a_test,b_test,pi_test)
    e=error_rate(dataset,tag_list,a_test,b_test,pi_test)

    with open (sys.argv[7],'wt') as pred_out:
        for i in a:
            for ind,n in enumerate(i):
                if ind == len(i)-1:
                    pred_out.write(str(n))
                else:
                    pred_out.write(str(n)+' ')
            pred_out.write('\n')

    with open (sys.argv[8],'w') as metric_out:
        metric_out.write("Average Log-Likelihood: "+str(log)+'\n')
        metric_out.write("Accuracy: "+str(e))
        
