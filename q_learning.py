from environment import MountainCar
import numpy as np
import sys

def findQ_max (state, weight, bias, epsilon):
    list_r = [0,0,0]
    for i in [0,1,2]:
        list_r[i] = np.dot(state.T, weight[:,i].reshape((len(weight),1))).item()+bias        
    max_val = max(list_r)
    max_act = np.argmax(list_r) 
    if epsilon == 0:
        return max_val, max_act
    elif epsilon !=0:
        if np.random.random()>epsilon:
            return max_val, max_act
        else:
            act = np.random.choice([0,1,2])
            val = np.dot(state.T, weight[:,act].reshape((len(weight),1))).item()+bias 
            return val, act
            
def vectorize_tile (t_state,rep):
    if rep == 'tile':
        weight_m = np.zeros((2048,1))
    elif rep == 'raw':
        weight_m = np.zeros((2,1))
        
    for k, v in t_state.items():
        weight_m[k]=v
    return weight_m

def para_init (rep):
    bias = 0.0
    if rep == 'tile':
        weight = np.zeros((2048,3))
    elif rep == 'raw':
        weight = np.zeros((2,3))
    return weight, bias

def weight_deriv (state, action,rep):
    if rep == 'tile':
        de_weight = np.zeros((2048,3))
        de_weight.T[action]=state.T
    elif rep == 'raw':
        de_weight = np.zeros((2,3))
        de_weight.T[action]=state.T
    return de_weight

def optimal_update (weight, bias, alpha, gamma, epsilon, environment,episode,iterat,rep):
    list_r = []
    for i in range(episode):
        total_r = 0
        curr_state = vectorize_tile(environment.reset(),rep)
        for t in range(iterat):
            curr_val, curr_act = findQ_max(curr_state, weight, bias,epsilon)
            info = environment.step(curr_act)
            future_state = vectorize_tile(info[0],rep)
            reward = info[1]
            total_r += reward
            goal = info[2]
            if goal == True:
                future_val, future_act = findQ_max(future_state,weight,bias,epsilon=0)
                weight = weight - alpha*(curr_val-(reward+gamma*future_val))*weight_deriv(curr_state,curr_act,rep)
                bias = bias - alpha*(curr_val-(reward+gamma*future_val))
                break
            else:
                future_val, future_act = findQ_max(future_state,weight,bias,epsilon=0)
                weight = weight - alpha*(curr_val-(reward+gamma*future_val))*weight_deriv(curr_state,curr_act,rep)
                bias = bias - alpha*(curr_val-(reward+gamma*future_val))
                curr_state = future_state
        list_r.append(total_r)
    return weight, bias, list_r
    

if __name__ == "__main__":
    env = MountainCar(mode=str(sys.argv[1]))
    we,b = para_init(str(sys.argv[1]))
    w,bi,r = optimal_update (we,b,float(sys.argv[8]),float(sys.argv[7]),float(sys.argv[6]),env,int(sys.argv[4]),int(sys.argv[5]),str(sys.argv[1]))
    with open (sys.argv[2], 'w') as weight_out:
        weight_out.write(str(bi)+'\n')
        for i in w:
            for it in i:
                weight_out.write(str(it)+'\n')
    with open (sys.argv[3], 'w') as return_out:
        for i in r:
            return_out.write(str(i)+'\n')
        
