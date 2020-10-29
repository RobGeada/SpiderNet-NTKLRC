import numpy as np
from sklearn.model_selection import train_test_split

def load_dolphin_data():
    train_xs = np.concatenate([np.load('data/NDD_Proc/train_xs.npy'), np.load('data/NDD_Proc/train_xs2_0.npy')])
    train_ys = np.concatenate([np.load('data/NDD_Proc/train_ys.npy'), np.load('data/NDD_Proc/train_ys2_0.npy')])
    train_idx = np.concatenate([np.load('data/NDD_Proc/train_idxs.npy'), np.load('data/NDD_Proc/train_idxs2_0.npy')])
    train_xs = np.transpose(train_xs, (0, 3, 1, 2))
    
    subset = np.random.choice(len(train_xs), 50000, replace=False)
    train_xs = train_xs[subset]
    train_ys = train_ys[subset]
    train_idx = train_idx[subset]
    
    MEAN = train_xs.mean((0,2,3))
    STD = train_xs.std((0,2,3))
    
    test_xs = np.concatenate([np.load('data/NDD_Proc/test_xs.npy'), np.load('data/NDD_Proc/test_xs2_0.npy')])
    test_ys = np.concatenate([np.load('data/NDD_Proc/test_ys.npy'), np.load('data/NDD_Proc/test_ys2_0.npy')])
    test_idx = np.concatenate([np.load('data/NDD_Proc/test_idxs.npy'), np.load('data/NDD_Proc/test_idxs2_0.npy')])
    test_xs = np.transpose(test_xs, (0, 3, 1, 2))
    
    subset = np.random.choice(len(test_xs), 10000, replace=False)
    test_xs = test_xs[subset]
    test_ys = test_ys[subset]
    test_idx = test_idx[subset]
    
    return [train_xs, train_idx, train_ys], [test_xs, test_idx, test_ys], MEAN, STD
    