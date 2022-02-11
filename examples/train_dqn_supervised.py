from tqdm import trange
from tqdm.autonotebook import tqdm
import dipy.reconst.dti as dti
from dipy.tracking import utils

import time 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import sys
sys.path.insert(0,'..')

import dfibert.envs.RLTractEnvironment_fast as RLTe
from dfibert.tracker.nn.rl import DQN
from dfibert.envs._state import TractographyState
from dfibert.tracker import save_streamlines


class SupervisedRewardDataset(Dataset):
    def __init__(self, inp, outp):
        self.inp = inp
        self.outp = outp
        
    def __getitem__(self, index):
        return (self.inp[index,], self.outp[index,])
    
    def __len__(self):
        return len(self.inp)


def save_model(path_checkpoint, model, epoch, loss, n_actions):
    print("Writing checkpoint to %s" % (path_checkpoint))
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["loss"] = loss
    checkpoint["n_actions"] = n_actions
    torch.save(checkpoint, path_checkpoint)


#------------------
def train():  
    batch_size = 1024
    seed_selection_fa_Threshold = 0.1
    epochs = 1000
    lr = 1e-4


    #------------------
    seeds_CST = np.load('data/ismrm_seeds_CST.npy')
    seeds_CST = torch.from_numpy(seeds_CST)
    env = RLTe.RLTractEnvironment(dataset = 'ISMRM', step_width=0.8,
                                  device = 'cuda:0', seeds = seeds_CST, action_space=20,
                                  odf_mode = "DTI", 
                                  fa_threshold=0.2, tracking_in_RAS=False)


    #------------------
    # initialize points for training
    dti_model = dti.TensorModel(env.dataset.gtab, fit_method='LS')
    dti_fit = dti_model.fit(env.dataset.dwi, mask=env.dataset.binary_mask)

    fa_img = dti_fit.fa
    seed_mask = fa_img.copy()
    seed_mask[seed_mask >= seed_selection_fa_Threshold] = 1
    seed_mask[seed_mask < seed_selection_fa_Threshold] = 0

    seeds = utils.seeds_from_mask(seed_mask, affine=np.eye(4), density=1)  # tracking in IJK
    seeds = torch.from_numpy(seeds).to(env.device)
    print("We got %d seeds" % (len(seeds)))

    #------------------
    # uniformly sample a point within our brain mask

    noActions = 20
    dimDWI = 100
    noPoints = len(seeds)

    dwi_data = torch.zeros([noPoints,3,3,3,dimDWI], device = env.device)
    rewards = torch.zeros([noPoints,noActions], device = env.device)

    for i in tqdm(range(noPoints), ascii=True):
        pos = seeds[i]

        # instantiate TractographyState
        state = TractographyState(pos, None)

        # call env.reward_for_state
        reward_ = env.reward_for_state(state, direction = "forward", prev_direction = None)

        #dwi_ = env.dwi_interpolator(pos.to(env.device))
        dwi_interpol_ = env.interpolate_dwi_at_state(pos.to(env.device))

        dwi_data[i,:,:,:,:] = dwi_interpol_
        rewards[i,:] = reward_


    ds = SupervisedRewardDataset(dwi_data, rewards)
    train_loader = DataLoader(ds, batch_size=batch_size,shuffle=True)


    model = DQN(input_shape = 3*3*3*dimDWI, n_actions=noActions, hidden_size=128, num_hidden=3).to(env.device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.MSELoss()

    begin = time.time()
    with trange(epochs, unit="epochs", ascii=True) as pbar:
        for epoch in pbar:
            # Set current and total loss value
            current_loss = 0.0

            model.train()   # Optional when not using Model Specific layer
            for i, data in enumerate(train_loader,0):
                x_batch = data[0].view([data[0].shape[0], -1])
                y_batch = data[1]
                optimizer.zero_grad()
                pred = model(x_batch)

                loss = criterion(pred,y_batch)

                loss.backward()
                optimizer.step()

            if( (epoch % 50) == 0):
                p_cp = 'checkpoints/defi_super_%d.pt' % (epoch)
                save_model(p_cp, model, epoch, loss.item(), noActions)

            pbar.set_postfix(loss=loss.item())

    end = time.time()
    print("time:", end - begin)




if __name__ == "__main__":
    train()