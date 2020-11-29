import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Lambda, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt



def get_model(layers, mode):
    """
    Returns Sequential model based on layers and mode
    Parameters:
        layers: List of (num_neurons, activation, dropout)
        mode:   str, train or eval. Train has dropout, test does not
    Returns:
        A sequential class object
    """
    model=Sequential()
    for i, layer_data in enumerate(layers):
        if layer_data[0]!='Lambda' and layer_data[0]!='lambda':
            num_nodes, l_activation, l_dropout=layer_data
            model.add(Dense(num_nodes, activation=l_activation))
            if mode=='Train' or mode=='train':
                model.add(Dropout(l_dropout))
        else:
            lamb_func_=layer_data[1]
            num_nodes_prev=layers[i-1][0]
            model.add(Lambda(lamb_func_, output_shape=(None, num_nodes_prev)))
        if i<=4:
            model.add(LayerNormalization())
    model.add(Dense(1))
    return model


def custom_loss(gamma):
    """
    Wrapper Custom loss function
    Parameters:
        gamma: The discounting parameter.
    Returns:
        loss function
    """
    def cl(y_true, y_pred):
        reward=y_true[:,0]
        target=y_true[:,1]
        return keras.backend.mean(keras.backend.abs(reward+gamma*target-y_pred))
    
    return cl

def schedule(epoch, lr):
    """
    The learning rate schedule
    Parameters:
        epoch and current learning rate
    Returns:
        lr: updated learning rate
    """
    if epoch==15:
        lr=lr/4
    return lr

def get_threshold(train_step):

    """
    Returns p (in (0,1)) such that optimal action taken with probability p. 
    The policy is Greedy in the limit of infinity.
    Parameters:
        train_step: The iteration index
    Returns:
        p: scaler
    """

    x=float(train_step)
    if x==0:
        return 0
    x=x/5
    return float(x)/(1+float(x))

def custom_scaler(train_x, scaler):
    """
    Scales data by simply dividing. No subtraction takes place.
    Parameters:
        train_x: The data. Shape=(m,n)
        scaler: The scaling vector. Shape= (m,)
    Returns:
        train_x: Inplace scaling happens.
    """
    train_x=train_x/scaler
    return train_x
    

def get_opt_action(car, network, actions, data_scaler, usescaler):
    """
    Given the car, network and bool usescaler, this function returns the best action
    and reward for each action for current state of car.
    Parameters:
        car: The car object. Used to decipher current state.
        network: The model that takes in state and action and returns the (estimated) long term reward.
        actions: Allowed actions as an numpy array.
        data_scaler: Numpy vector that is used for scaling using custom scaler
        usescaler: bool. True iff network is to be fed data scaled by data_scaler
    Returns:
        action and list: reward for each action
    """
    rows=len(actions)
    state_vector=np.zeros((9, ))
    state_vector[:8]=car.sensors
    state_vector[8]=car.speed
    m1=np.ones((rows, 1))
    net_input=np.matmul(m1, state_vector.reshape((1, -1)))
    net_input=np.hstack((net_input, actions))
    if usescaler:
        net_input=custom_scaler(net_input, data_scaler)
    res=network.predict(net_input)
    best_action_idx=np.argmax(res)
    return actions[best_action_idx]

    
def update_replay_helper(states, network, actions, data_scaler, usescaler):
    """
    For each row in 'states', return the max long term reward according to 'network'.
    Parameters:
        states: states is an N x 9 array, having, well, N states.
        network: The model that takes in state and action and returns the (estimated) long term reward.
        actions: Allowed actions as an numpy array.
        data scaler: is an array of size 12 (number of features fed to the network).
                     12=9 (i.e. the state) + 3(i.e. the action). It only scales, so no subtracting.
        usescaler: Bool.
    Returns:
        max_value: An Nx1 numpy array having max reward from each state.
    """
    rows=len(states)
    m1=np.ones((rows, 1))
    max_val=np.full((rows, 1), -np.inf)
    for action_idx in range(len(actions)):
        temp=np.matmul(m1, actions[action_idx].reshape((1, -1)))
        net_input=np.hstack((states, temp))
        if usescaler:
            net_input=custom_scaler(net_input, data_scaler)
        max_val=np.maximum(max_val, network.predict(net_input))
    return max_val


def populate_replay_memory(car, network, actions, replay_memory, insert_index,
                          rows_to_fill, data_scaler, usescaler, train_step, exploration, max_force,
                          **kwargs):
    """
    Populates the replay memory- a dataset of s,a,s`,Q. This must always be followed a call to update_replay_memory.
    Parameters:
        car:
        netwrok:
        actions:
        replay_memory: The state at time t will be sensors, speed, car_angle. So 10 values.
                    The action at time t will be 2 values, the throttle and steering.
                    At t+1, state has 10 values.
                    The reward at time t will be speed*(speed_multiplier) or penalty if off track. So 1 value.
                    The target reward is 1 value.
                    
                    So, replay memory needs 24 columns.

                    During training, we only supply the first 12 columns, except the 10th (car-angle), to the network.
                    The last 2 columns are supplied in y_true, and are used in loss computation.
                    
                    The last 80 rows of replay_memory contain data (s,a,s`,r) from early episodes
        inset_index:
        rows_to_fill:
        data_scaler:
        usescaler:
        train_step:
        exploration:
        max_force:
        
    Returns:
        None. Inplace populates the replay_memory, with rows_to_fill new rows filled consecutively starting from insert_index. 

    """
    
    if np.random.choice([0,0,0,0,0,1]):
        car.re_spawn()
    optimalActionProb=get_threshold(train_step)
    time_map=[]
    for i in range(rows_to_fill):
        
        row_data=np.zeros((1, 24))
        row_data[0, :8]=row_data[0, :8]+car.sensors
        row_data[0, 8]=car.speed
        row_data[0, 9]=car.car_angle
        
        # choose an action
        x=np.random.uniform(0, 1)
        if x>optimalActionProb:
            throttle, steer=actions[np.random.choice(range(len(actions)))]
        else:
            throttle, steer=get_opt_action(car, network, actions, data_scaler, usescaler)
        
        if 'disable_throttle' in kwargs and kwargs['disable_throttle']:
            throttle=0
            car.speed=kwargs['def_speed']
        
        row_data[0, 10:12]=[throttle, steer]
        
        theta=car.car_angle
        fin_f1, fin_f2=throttle*np.sin(theta)-steer*np.cos(theta), throttle*np.cos(theta)+steer*np.sin(theta)
        car.execute_forces(fin_f1, fin_f2, max_magnitudes=max_force)
        
        # update exploration
        exploration[car.integer_position_[0], car.integer_position_[1]]=1
        
        
        row_data[0, 12:20]=row_data[0, 12:20]+car.sensors
        row_data[0, 20]=car.speed
        row_data[0, 21]=car.car_angle
        
        # calculate reward
        if car.collided_on_last:
            rew=-500
        else:
            rew=5-car.time_in_current_sector/20+3*car.speed/car.max_speed
        time_map.append(car.time_in_current_sector)
            
            
        row_data[0, 22]=rew
        row_data[0, 23]=0
        replay_memory[insert_index[0]]=replay_memory[insert_index[0]]*0+row_data
        if train_step<5:
            insert_index[0]=(insert_index[0]+1)%len(replay_memory)
        else:
            insert_index[0]=(insert_index[0]+1)%(len(replay_memory)-80)
        
    plt.plot(time_map)
    plt.show()
    return


def update_replay_memory(target_model, actions, replay_memory,
                         data_scaler, usescaler):
    """
    Updates the 'target value' column of the replay memory.
    Parameters:
        target_model:
        actions:
        replay_memory:
        data_scaler:
        usescaler:
    Returns:
        None. Inplace updates replay memory's target value column.
    """
    # select the next state columns, except the car angle
    l=[i for i in range(12, 12+8+1)]
    temp=replay_memory[:, l]
    
    # get the new target column that has max Q for each state
    res=update_replay_helper(temp, target_model, actions, data_scaler, usescaler)
    replay_memory[:, 23]=res.reshape((len(replay_memory), ))
    
    return

