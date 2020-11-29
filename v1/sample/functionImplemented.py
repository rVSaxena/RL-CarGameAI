import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Lambda, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, EarlyStopping
import numpy as np



def get_model(layers, mode):
    """
    Returns Sequential data based on layers and mode
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
            if mode=='Train':
                model.add(Dropout(l_dropout))
        else:
            lamb_func_=layer_data[1]
            num_nodes_prev=layers[i-1][0]
            model.add(Lambda(lamb_func_, output_shape=(None, num_nodes_prev)))
        if i<=4:
            model.add(LayerNormalization())
    model.add(Dense(1))
    return model


def custom_loss(y_true, y_pred):
    """
    Custom loss function
    Parameters:
        y_true, y_pred as usual
    Returns:
        loss(y_true, y_pred)
    """
    reward=y_true[:,0]
    target=y_true[:,1]
    return keras.backend.mean(keras.backend.abs(reward+0.8*target-y_pred))

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
    Returns p (in (0,1))  such that optimal action taken with probability p. 
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
    Given the car, network and bool usescaler, this function returns the best action and reward for each action for current state of car.
    Parameters:
        car: The car object. Used to decipher current state.
        network: The model that takes in state and action (1-h encoded) and returns the (estimated) long term reward.
        actions: allowed actions as an numpy array.
        data_scaler: Numpy vector that is used for scaling using custom scaler
        usescaler: bool. True iff network is to be fed data scaled by data_scaler
    Returns:
        action and list: reward for each action
    """
    m1=np.ones((3, 1))
    state=np.zeros((1, 9))
    state[0, :8]=car.sensors
    state[0, 8]=car.speed
    test_x=np.hstack((np.matmul(m1, state), np.eye(3)))
    if usescaler:
        test_x=custom_scaler(test_x, data_scaler)
    res=network.predict(test_x)
    best_action=np.argmax(res)
    if best_action==0:
        return -1, res
    if best_action==1:
        return 0, res
    if best_action==2:
        return 1, res

    
def update_replay_helper(states, network, actions, data_scaler, usescaler):
    """
    For each row in 'states', return the max long term reward according to 'network'.
    Parameters:
        states: states is an N x 9 array, having, well, N states.
        network: The model that takes in state and action (1-h encoded) and returns the (estimated) long term reward.
        actions: Allowed actions as an numpy array.
        data scaler: is an array of size 12 (number of features fed to the network).
                     12=9 (i.e. the state) + 3(i.e. the action). It only scales, so no subtracting.
        usescaler: Bool.
    Returns:
        max_value: An Nx1 numpy array having max reward ffrom each state.
    """
    rows=len(states)
    m1=np.ones((rows, 1))
    one_hot_actions=np.eye(len(actions))
    
    max_value=np.full((rows, 1), -np.inf)
    
    for action_index in range(len(one_hot_actions)):
        temp=np.matmul(m1, one_hot_actions[action_index].reshape((1, -1)))
        network_input=np.hstack((states, temp))
        if usescaler:
            network_input=custom_scaler(network_input, data_scaler)
        max_value=np.maximum(max_value, network.predict(network_input))
        
    return max_value


def populate_replay_memory(car, network, actions, replay_memory, insert_index,
                          rows_to_fill, data_scaler, usescaler, train_step, exploration):
    """
    Populates the replay memory- a dataset of s,a,s`,Q.
    Parameters:
        car:
        netwrok:
        actions:
        replay_memory: The state at time t will be sensors, speed, car_angle. So 10 values.
                    The action at time t will be a value, but it is encoded 1-h. Since 3 possible actions, hence 3 array units.
                    At t+1, state has 10 values.
                    The reward at time t will be speed*(speed_multiplier) or penalty if off track. So 1 value.
                    The target reward is 1 value.
                    
                    So, replay memory needs 25 columns.

                    During training, we only supply the first 13 columns, except the 10th (car-angle), to the network.
                    The last 2 columns are supplied in y_true, and are used in loss computation.
        inset_index:
        rows_to_fill:
        data_scaler:
        usescaler:
        train_step:
        exploration:
        
    Returns:
        None. Inplace populates the replay_memory, with rows_to_fill new rows filled consecutively starting from insert_index. 

    """
    
    if np.random.choice([0, 1]):
        car.re_spawn()
        
    for i in range(rows_to_fill):
        
        row_data=np.zeros((1, 25))
        row_data[0, :8]=row_data[0, :8]+car.sensors
        row_data[0, 8]=car.speed
        row_data[0, 9]=car.car_angle
        
        # choose an action
        x=np.random.uniform(0, 1)
        if x>get_threshold(train_step):
            direction=actions[np.random.choice(range(len(actions)))]
        else:
            direction, best_action_value=get_opt_action(car, network, actions, data_scaler, usescaler)
        
        # prepare action
        if direction==0:
            one_hot_direction=[0, 1, 0]
        elif direction==-1:
            one_hot_direction=[1, 0, 0]
        else:
            one_hot_direction=[0, 0, 1]
        row_data[0, 10:13]=one_hot_direction
        old_angle=car.car_angle
        theta=car.car_angle
        if direction==-1:
            theta=np.mod(theta+np.pi/2, 2*np.pi)
        elif direction==1:
            theta=np.mod(theta-np.pi/2, 2*np.pi)
            
        # execute it
        fin_f1, fin_f2=np.sin(theta), np.cos(theta)
        car.execute_forces(fin_f1, fin_f2, max_magnitudes=500)
        
        # update exploration
        exploration[car.integer_position_[0], car.integer_position_[1]]=1
        
        
        row_data[0, 13:21]=row_data[0, 13:21]+car.sensors
        row_data[0, 21]=car.speed
        row_data[0, 22]=car.car_angle
        
        # calculate reward
        if car.collided_on_last:
            rew=-500
        else:
            rew=car.speed/10
            
            
        row_data[0, 23]=rew
        row_data[0, 24]=0
        replay_memory[insert_index[0]]=replay_memory[insert_index[0]]*0+row_data
        insert_index[0]=(insert_index[0]+1)%len(replay_memory)
        
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
    l=[i for i in range(13, 13+8+2) if i!=(13+7+2)]
    temp=replay_memory[:, l]
    
    # get the new target column that has max Q for each state
    res=update_replay_helper(temp, target_model, actions, data_scaler, usescaler)
    replay_memory[:, 24]=res.reshape((len(replay_memory), ))
    
    return

