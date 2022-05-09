from tqdm import trange, tqdm
import dipy.reconst.dti as dti
from dipy.tracking import utils

import time 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import sys
sys.path.insert(0,'..')

import dfibert.envs.RLTractEnvironment_fast as RLTe
from dfibert.envs._state import TractographyState
from dfibert.tracker import save_streamlines

from copy import deepcopy
import os

from dfibert.tracker.nn.mlp import MLP

from dfibert.data import DataPreprocessor

def getEllasData(pData = "data/Ella", b0_threshold = 1000):
    preprocessor = DataPreprocessor().normalize().crop(1000).fa_estimate() # 
    file_mapping = {'bvals': 'bvals', 'bvecs': 'bvecs',
                        'img': 'data.nii', 't1': 'T1w_acpc_dc_restore_1.25.nii', 'mask': 'nodif_brain_mask.nii'}
    dataset = preprocessor._get_from_file_mapping(pData, file_mapping)
    return dataset


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


directions = ["forward", "backward"]

#------------------
def train(env, batch_size: int = 1024, epochs: int = 1000, lr: float = 1e-4, 
          seed_selection_fa_Threshold: float = 0.1,
          path: str = './supervised_checkpoints/',
          wandb_log: bool = False):

    os.makedirs(path, exist_ok=True)

    #------------------
    # initialize points for training
    dti_model = dti.TensorModel(env.dataset.gtab, fit_method='LS')
    dti_fit = dti_model.fit(env.dataset.dwi, mask=env.dataset.binary_mask)

    fa_img = dti_fit.fa
    seed_mask = fa_img.copy()
    seed_mask[seed_mask >= seed_selection_fa_Threshold] = 1
    seed_mask[seed_mask < seed_selection_fa_Threshold] = 0

    seeds = utils.seeds_from_mask(seed_mask, affine=np.eye(4), density=1)  # tracking in IJK
    #np.save(path + 'seeds', seeds) # optinally save the seeds for later use
    seeds = torch.from_numpy(seeds).to(env.device)
    env.seeds = seeds   #  after training and the first tracking, you probably don't need to overwrite the seeds
    print("We got %d seeds" % (len(seeds)))

    #------------------
    # uniformly sample a point within our brain mask

    noActions = env.action_space.n
    _ = env.reset()
    stateShape = tuple(env.state.getValue().shape)
    noPoints = len(seeds)

#-> bei der folgenden Zeile stürtzt das Programm mit einem Kill ab
    dwi_data = torch.zeros([noPoints,*stateShape], device = env.device)
    rewards = torch.zeros([noPoints, noActions], device = env.device)

    print("Filling dataset..")
    for i in tqdm(range(noPoints), ascii=True):
        pos = seeds[i]
        # instantiate TractographyState
        state = TractographyState(pos, None)

        # call env.reward_for_state
        reward_ = env.reward_for_state(state, direction ='forward', prev_direction = None)
        dwi_interpol_ = env.interpolate_dwi_at_state(pos.to(env.device))

        dwi_data[i,:,:,:,:] = dwi_interpol_
        rewards[i, :] = reward_


    print("..done!")
    ds = SupervisedRewardDataset(dwi_data, rewards)
    train_loader = DataLoader(ds, batch_size=batch_size,shuffle=True)

    model = MLP(input_size=np.prod(stateShape), output_size=noActions, hidden_size=1024, num_hidden=8, activation=torch.nn.LeakyReLU()).to(env.device)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.MSELoss() # you can test several other loss functions
    #criterion = torch.nn.L1Loss()

    print("Start pretraining DQN..")
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

                #print("---Debugging---")
                #print(x_batch.shape, y_batch.shape)

                loss = criterion(pred,y_batch)
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


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pData = f"/Users/niccay/Code/miccai_paper/deepFibreTracking/examples/data/Ella/"
    b0_threshold = 1000

    dataset_ella = getEllasData(pData = pData, b0_threshold = b0_threshold)
    
    # change the dataset parameter to the path of your dataset folder
    # odf_mode = "DTI" for quick tests but less accurate results
    # odf_mode = "CSD" for accurate results, odf computation takes ~15 minutes
   
    env = RLTe.RLTractEnvironment(step_width=0.8, dataset = dataset_ella, neuroanatomical_reward = False,
                              device = device, seeds = None, tracking_in_RAS = False,
                              odf_state = False, odf_mode = "DTI")

    # if you want to load the odf data
    # move to the RLTractEnvironment_fast.py file
    # and uncomment the line 168
    # change the paths in line 168 and 107 to match accordingly to you paths
    # set load_odf = True above as soon as you have saved one odf file

    noActions = env.action_space.n
    _ = env.reset()
    stateShape = tuple(env.state.getValue().shape)
    model = MLP(input_size=np.prod(stateShape), output_size=noActions, hidden_size=1024, num_hidden=8, activation=torch.nn.LeakyReLU()).to(env.device)
    
    # load an already trained model from disk
    #model.load_state_dict(torch.load("pretrained_test.pt", map_location=env.device))

    # train a new model
    model = train(env, seed_selection_fa_Threshold = 0.3)
    torch.save(model.state_dict(), "pretrained_test.pt")
   
    # declare the "agent" function
    agent = lambda state: torch.argmax(model(state))

    # load saved seeds from disk
    #seeds = np.load('./supervised_checkpoints/seeds.npy')
    
    # overwrite the seeds of the environment if needed
    #env.seeds = torch.from_numpy(seeds).to(env.device)
    
    # compute the tractogram on a lower number of seeds for quick tests
    #env.seeds = env.seeds[:1000]

    # compute a "ground truth" tractogram, the ML model is only capable
    # to learn to copy this artificial ground truth
    #gt_streamlines = env.track()

    # compute the learned tractogram
    streamlines = env.track(agent)

    # save the tractogram to a vtk file
    #save_streamlines(gt_streamlines, "gt_streamlines.vtk")
    save_streamlines(streamlines, "test_tractogram.vtk")
