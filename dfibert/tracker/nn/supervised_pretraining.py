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
from dfibert.envs._state import TractographyState
from dfibert.tracker import save_streamlines

from copy import deepcopy
import os

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
def train(dqn, env, batch_size: int = 1024, epochs: int = 1000, lr: float = 1e-4, 
          seed_selection_fa_Threshold: float = 0.1,
          path: str = './supervised_checkpoints',
          wandb_log: bool = False):

    os.makedirs(path, exist_ok=True)

    if wandb_log:
        import wandb 

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

    noActions = env.action_space.n
    #dimDWI = 100
    _ = env.reset()
    stateShape = tuple(env.state.getValue().shape)
    noPoints = len(seeds)

    #dwi_data = torch.zeros([noPoints,3,3,3,dimDWI], device = env.device)
    dwi_data = torch.zeros([noPoints,*stateShape], device = env.device)
    actions = torch.zeros([noPoints], device = env.device, dtype=torch.int64)

    print("Filling dataset..")
    for i in tqdm(range(noPoints), ascii=True):
        pos = seeds[i]
        # instantiate TractographyState
        state = TractographyState(pos, None)

        # call env.reward_for_state
        reward_ = env.reward_for_state(state, direction = "forward", prev_direction = None)
        action_ = torch.argmax(reward_)
        #action_ = torch.LongTensor(action_)
        #dwi_ = env.dwi_interpolator(pos.to(env.device))
        dwi_interpol_ = env.interpolate_dwi_at_state(pos.to(env.device))

        dwi_data[i,:,:,:,:] = dwi_interpol_
        #rewards[i,:] = reward_
        actions[i] = action_

    print("..done!")
    ds = SupervisedRewardDataset(dwi_data, actions)#rewards)
    train_loader = DataLoader(ds, batch_size=batch_size,shuffle=True)

    model = deepcopy(dqn)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()

    print("Start pretraining DQN..")
    begin = time.time()
    with trange(epochs, unit="epochs", ascii=True) as pbar:
        for epoch in pbar:
            # Set current and total loss value
            current_loss = 0.0

            model.train()   # Optional when not using Model Specific layer
            for i, data in enumerate(train_loader,0):
                x_batch = data[0].view([data[0].shape[0], -1])
                y_batch = data[1].type(torch.int64)
                optimizer.zero_grad()
                pred = model(x_batch)

                loss = criterion(pred,y_batch)
                if wandb_log:
                    wandb.log({"Pretraining: supervised loss": loss})
                loss.backward()
                optimizer.step()

                if loss < 0.005:
                    break

            if( loss < 0.0005):
                print("Early stop at loss %.4f" % (loss))
                p_cp = path+'defi_super_%d.pt' % (epoch)
                save_model(p_cp, model, epoch, loss.item(), noActions)
                break
            if( (epoch % 50) == 0):
                p_cp = path+'defi_super_%d.pt' % (epoch)
                save_model(p_cp, model, epoch, loss.item(), noActions)

            pbar.set_postfix(loss=loss.item())

    end = time.time()
    print("..done! time:", end - begin)
    return deepcopy(model)
