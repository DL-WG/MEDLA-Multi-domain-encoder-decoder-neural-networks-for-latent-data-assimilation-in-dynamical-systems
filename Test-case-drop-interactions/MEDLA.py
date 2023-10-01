import os
import glob
import time
from random import randint
import copy
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class AlignDataset(Dataset):
    def __init__(self, map_arr: np.ndarray):
        # example map entry array(['extract_arr/test/03_06_C10/frame0170.npy',
        #                    'extract_arr/NC_CFD_pressure/frame0171.npy'], dtype='<U47')
        #self.path_to_data = path_to_data
        #self._config = config
        self.map_arr = map_arr
            
    def __len__(self):
        return self.map_arr.shape[0]
    
    def __getitem__(self, idx):
        # return input for encoder1, encoder2
        exp_img = self.load_img(self.map_arr[idx,0])
        cfd_img = self.normalize(self.load_img(self.map_arr[idx,1]))
        exp_img = exp_img.reshape(1,exp_img.shape[0],exp_img.shape[1])
        cfd_img = cfd_img.reshape(1,cfd_img.shape[0],cfd_img.shape[1])
        return torch.from_numpy(exp_img).type(torch.FloatTensor), torch.from_numpy(cfd_img).type(torch.FloatTensor)
        
    def load_img(self, path: str) -> np.ndarray:
        return np.load(path)
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        return (img - np.min(img))/(np.max(img) - np.min(img))

class ExpDataset(Dataset):
    def __init__(self, map_arr: np.ndarray):
        # example map entry array(['extract_arr/test/03_06_C10/frame0170.npy',
        #                    'extract_arr/NC_CFD_pressure/frame0171.npy'], dtype='<U47')
        #self.path_to_data = path_to_data
        #self._config = config
        self.map_arr = map_arr
            
    def __len__(self):
        return self.map_arr.shape[0]
    
    def __getitem__(self, idx):
        # return input for encoder1
        exp_img = self.load_img(self.map_arr[idx])
        exp_img = exp_img.reshape(1,exp_img.shape[0],exp_img.shape[1])
        return torch.from_numpy(exp_img).type(torch.FloatTensor)
        
    def load_img(self, path: str) -> np.ndarray:
        return np.load(path)

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SubsetRandomSampler(Sampler):
    def __init__(self, num_samples, num_samples_per_epoch):
        self.num_samples = num_samples
        self.num_samples_per_epoch = num_samples_per_epoch

    def __iter__(self):
        return iter(torch.randperm(self.num_samples)[:self.num_samples_per_epoch])

    def __len__(self):
        return self.num_samples_per_epoch

class encoder(nn.Module):
    def __init__(self, *, chan: list, n_layer: int, kernel_size: list,
                latent_dim: int, pool_list: list, flatten_len: int, act_fn: object = nn.ReLU()):
        """
        Inputs:
            - ini_chan: Numer of channels at the 1st layer
            - n_layer: Number of layer of the autoencoder
            - kernel_size: Size of the kernel (square)
            - latent_dim: Size of the latent dimension
            - pool_size: Size of the maxpooling (square)
        """
            
        super().__init__()
        layers=[]
        for i in range(n_layer):
            if i == 0:
                prev_chan = 1
            else:
                prev_chan = chan[i-1]
            layers.append(nn.Conv2d(prev_chan, chan[i], kernel_size=kernel_size[i], padding = 'same', stride = 1))
            layers.append(nn.BatchNorm2d(chan[i]))
            layers.append(nn.MaxPool2d(kernel_size = pool_list[i]))
            layers.append(act_fn())
        layers.append(nn.Flatten())
        #layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(flatten_len, latent_dim))
        #layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def test(self, dim: int):
        input = torch.arange(1, dim**2+1, dtype=torch.float32).view(1, 1, dim, dim)
        print(self.forward(input.to(device)).shape)

class cfd_encoder(nn.Module):
    def __init__(self, *, chan: list, n_layer: int, kernel_size: list,
                latent_dim: int, pool_list: list, flatten_len: int, act_fn: object = nn.ReLU()):
        """
        Inputs:
            - ini_chan: Numer of channels at the 1st layer
            - n_layer: Number of layer of the autoencoder
            - kernel_size: Size of the kernel (square)
            - latent_dim: Size of the latent dimension
            - pool_size: Size of the maxpooling (square)
        """
            
        super().__init__()
        layers=[]
        for i in range(n_layer):
            if i == 0:
                prev_chan = 1
            else:
                prev_chan = chan[i-1]
            layers.append(nn.Conv2d(prev_chan, chan[i], kernel_size=kernel_size[i], padding = 'same', stride = 1))
            layers.append(nn.BatchNorm2d(chan[i]))
            layers.append(nn.MaxPool2d(kernel_size = pool_list[i]))
            layers.append(act_fn())
        layers.append(nn.Flatten())
        #layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(flatten_len, 256))
        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 64))
        layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, latent_dim))
        #layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    def test(self, dim: int):
        input = torch.arange(1, dim**2+1, dtype=torch.float32).view(1, 1, dim, dim)
        print(self.forward(input.to(device)).shape)

class decoder(nn.Module):
    def __init__(self, *, chan: list, n_layer: int, kernel_size: int,
                latent_dim: int, pool_list: list, flatten_len: int, act_fn: object = nn.ReLU()):
        """
        Inputs:
            - ini_chan: Numer of channels of the last layer of encoder
            - n_layer: Number of layer of the autoencoder
            - kernel_size: Size of the kernel (square)
            - latent_dim: Size of the latent dimension
            - pool_size: Size of the upsampling (square)
        """
        super().__init__()
        layers=[]
        self.latent_dim = latent_dim
        self.chan = chan
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, flatten_len),
            #act_fn()
        )
        for i in range(n_layer):
            if i == 0:
                prev_chan = chan[0]
            else:
                prev_chan = chan[i-1]
            layers.append(nn.Conv2d(prev_chan, chan[i], kernel_size=kernel_size[i], padding = 'same', stride = 1))
            layers.append(act_fn())
            layers.append(nn.UpsamplingNearest2d(scale_factor = pool_list[i]))
        layers.append(nn.Conv2d(chan[-1], 1, kernel_size=kernel_size[-1], padding = 'same', stride = 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        # input shape (X,1,1,16)
        x = self.linear(x)
        l = int(np.sqrt(self.latent_dim))
        x = x.reshape(x.shape[0], -1, l, l)
        return self.net(x)
    
    def test(self):
        input = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 1, 16)
        print(self.forward(input).shape)

class MEDLA(nn.Module):
    def __init__(self, *, en1_chan_list: list, en2_chan_list: list, en1_pool_list: list,
                 en1_ker_list: list, en2_pool_list: list, en2_ker_list: list):
        super().__init__()
        self.en1_chan_list = en1_chan_list
        self.de_chan_list = list(reversed(en1_chan_list))
        self.en2_chan_list = en2_chan_list
        self.en1_pool_list = en1_pool_list
        self.de_pool_list = list(reversed(en1_pool_list))
        self.en1_ker_list = en1_ker_list
        self.de_ker_list = list(reversed(en1_ker_list))
        self.en2_pool_list = list(en2_pool_list)
        self.en2_ker_list = en2_ker_list
        
        # Exp encoder
        self.encoder1 = encoder(chan = self.en1_chan_list, n_layer=3, kernel_size=self.en1_ker_list, 
                       latent_dim = 16, pool_list = self.en1_pool_list, flatten_len = 512, act_fn = nn.ReLU)
        # CFD encoder
        self.encoder2 = cfd_encoder(chan = self.en2_chan_list, n_layer=3, kernel_size=self.en2_ker_list, 
                       latent_dim = 16, pool_list = self.en2_pool_list, flatten_len = 512, act_fn = nn.ReLU)
        # Shared decoder
        self.decoder = decoder(chan = self.de_chan_list, n_layer=3, kernel_size=self.de_ker_list, 
                       latent_dim = 16, pool_list = self.de_pool_list, flatten_len = 512, act_fn = nn.ReLU)
    
    def forward(self, x, mode: int):
        if mode == 1:
            x = self.encoder1(x)
        elif mode == 2:
            x = self.encoder2(x)
        else:
            print(mode, type(mode))
            raise ValueError
        return self.decoder(x)

class Trainer:
    def __init__(self, *, dataloader_dict: dict, model, lr: float, num_epoch: int):
        self.dataloader_dict = dataloader_dict
        self.model = model
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=50, min_lr=1E-6, verbose=True)
        self.best_wts = copy.deepcopy(self.model.state_dict())
        self.best_loss = np.inf
        self.num_epoch = num_epoch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")
        print(f"Using {self.device} device")
        print("PyTorch Version: ",torch.__version__)
        
        self.model.to(self.device)
    
    def _do_exp_epoch(self, phase: str):
        running_loss = 0.
        sample_num = len(self.dataloader_dict['exp'][phase].sampler)
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        for input_exp in self.dataloader_dict['exp'][phase]:
            input_exp = input_exp.to(self.device)
            # forward
            self.optimizer.zero_grad()
            outputs = self.model(input_exp, 1)
            loss = self.criterion(outputs, input_exp)
            if phase == "train":
                # backward
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()*input_exp.size(0)
        return running_loss/sample_num
    
    def _do_ft_epoch(self, phase: str):
        running_loss = 0.
        sample_num = len(self.dataloader_dict['align'][phase].sampler)
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        for input_exp, input_cfd in self.dataloader_dict['align'][phase]:
            input_exp = input_exp.to(self.device)
            input_cfd = input_cfd.to(self.device)
            # forward
            self.optimizer.zero_grad()
            #encoded_outputs = self.model.encoder1(input_exp)
            #encoded_cfd = self.model.encoder2(input_cfd)
            exp_outputs = self.model(input_exp, 1)
            cfd_outputs = self.model(input_cfd, 2)
            #loss = self.criterion(encoded_cfd, encoded_outputs)
            loss = self.criterion(cfd_outputs, exp_outputs)
            if phase == "train":
                # backward
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()*input_exp.size(0)
        return running_loss/sample_num
        
    def _do_align_epoch(self, phase):
        losses = {'exp': 0.0, 'align': 0.0}

        for mode in ['align', 'exp']:
            if phase =='train':
                self.model.train()  
            else:
                self.model.eval()  

            running_loss = 0.0
            if mode == 'exp':
                for input_exp in self.dataloader_dict[mode][phase]:
                    input_exp = input_exp.to(self.device)

                    self.optimizer.zero_grad() 
                    output = self.model(input_exp, 1)  

                    loss = self.criterion(output, input_exp)  
                    running_loss += loss.item()*inputs.size(0)
                    if phase == 'train':
                        loss.backward() 
                        self.optimizer.step() 
            else:
                for input_exp, input_cfd in self.dataloader_dict[mode][phase]:
                    input_exp = input_exp.to(self.device)
                    input_cfd = input_cfd.to(self.device)

                    for inputs, encoder_label in [(input_exp, 1), (input_cfd, 2)]:
                        # forward
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs, encoder_label)
                        loss = self.criterion(outputs, input_exp)
                        running_loss += 0.5*loss.item()*inputs.size(0)
                        if phase == "train":
                            # backward
                            loss.backward()
                            self.optimizer.step()

            avg_loss = running_loss / len(self.dataloader_dict[mode][phase].sampler)
            losses[mode] = avg_loss

        return losses
 
    
    def run_align(self):
        print("Running the Align phase\n")
        self.criterion = nn.BCELoss()
        for epoch in range(self.num_epoch):
            train_loss = self._do_align_epoch("train")
            with torch.no_grad():
                val_loss = self._do_align_epoch("val")
            avg_val_loss = val_loss['exp']+val_loss['align']
            self.scheduler.step(avg_val_loss)
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.best_wts = copy.deepcopy(self.model.state_dict())  
                torch.save(self.model.state_dict(), "MEDLA_best_align.pth")
                torch.save(self.model.state_dict(), "MEDLA_weight/best_align_"+str(epoch)+".pth")
            if epoch % 10 == 0:
                print(f"Exp Train loss: {train_loss['exp']:>7f}, Exp Validation loss: {val_loss['exp']:>7f} | Align Train loss: {train_loss['align']:>7f}, Align Validation loss: {val_loss['align']:>7f} | time: {time.strftime('%H:%M:%S')}")
            
    def run_exp(self):        
        print("Tuing Exp data\n")
        self.criterion = nn.BCELoss()
        self.model.load_state_dict(torch.load('MEDLA_best_align.pth'))
        tmp_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=50, min_lr=1E-8, verbose=True)
        self.best_loss = np.inf
        for epoch in range(self.num_epoch):
            train_loss = self._do_exp_epoch("train")
            with torch.no_grad():
                val_loss = self._do_exp_epoch("val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_wts = copy.deepcopy(self.model.state_dict())  
                torch.save(self.model.state_dict(), "MEDLA_exp.pth")
                torch.save(self.model.state_dict(), "MEDLA_weight/best_exp_"+str(epoch)+".pth")
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epoch}, train_exp_loss: {train_loss:>7f}, val_exp_loss: {val_loss:>7f} | time: {time.strftime('%H:%M:%S')}")      
                    
    def run_finetune(self):
        print("Running the finetune of latent space\n")
        #self.criterion = nn.MSELoss()
        self.criterion = nn.BCELoss()
        self.model.load_state_dict(torch.load('MEDLA_exp.pth'))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=50, min_lr=1E-6, verbose=True)
        self.best_loss = np.inf
                      
        for param in self.model.encoder1.parameters():
            param.requires_grad = False
        for param in self.model.decoder.parameters():
            param.requires_grad = False
        
        for epoch in range(self.num_epoch):
            train_loss = self._do_ft_epoch("train")
            with torch.no_grad():
                val_loss = self._do_ft_epoch("val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_wts = copy.deepcopy(self.model.state_dict())  
                torch.save(self.model.state_dict(), "MEDLA_BCE_ft.pth")
                #torch.save(self.model.state_dict(), "MEDLA_weight/best_ft_"+str(epoch)+".pth")
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epoch}, train_exp_loss: {train_loss:>7f}, val_exp_loss: {val_loss:>7f} | time: {time.strftime('%H:%M:%S')}") 


def split_data(train_map_array):
    num_array = train_map_array.shape[0]
    train_index = list(set(range(0,num_array,1))- set(range(0,num_array,5)))
    val_index = list(range(0,num_array,5))
    val_map_array = train_map_array[val_index]
    train_map_array  = train_map_array[train_index]

    return train_map_array, val_map_array

def get_dataloader_dict(batch_size):
    dataloader_dict = {}
    full_train_map_array, full_val_map_array = split_data(np.sort(np.asarray(glob.glob('extract_arr/train_down_sample/*/*')),axis=0))
    exp_train_map_array, exp_val_map_array = split_data(np.sort(np.asarray(glob.glob('extract_arr/test_down_sample/03_06*/*')),axis=0))
    align_train_map_array, align_val_map_array = split_data(np.load('align_map_arr.npy'))
                      
    for mode in ['exp', 'align']:
        if mode == 'exp':
            train_dataset = ExpDataset(full_train_map_array)
            val_dataset = ExpDataset(full_val_map_array)
        elif mode == 'align':
            train_dataset = AlignDataset(align_train_map_array)
            val_dataset = AlignDataset(align_val_map_array)
        
        if mode == 'exp':
            train_dataset_size = len(train_dataset)
            val_dataset_size = len(val_dataset)
            num_samples_per_epoch_train = int(195*4*0.8)
            num_samples_per_epoch_val = int(195*4*0.2)
            train_sampler = SubsetRandomSampler(train_dataset_size, num_samples_per_epoch_train)
            val_sampler = SubsetRandomSampler(val_dataset_size, num_samples_per_epoch_val)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler = train_sampler)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler = val_sampler)
        elif mode == 'align':
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        dataloader_dict[mode] = {'train': train_dataloader, 'val': val_dataloader}

    return dataloader_dict
                      
if __name__ == "__main__":
    seed_everything(0)
    print(torch.cuda.get_device_name(0))
    batch_size = 128
    dataloader_dict = get_dataloader_dict(batch_size)

    encoder_channel = [8, 16, 32] # Channels of encoder1, encoder2 are the same
    encoder_channel2 = [8, 16, 32]
    encoder1_kernel = [16, 8, 4]
    encoder2_kernel = [16, 8, 4]
    encoder1_pool = [4, 4, 4]
    encoder2_pool = [4, 4, 4]

    MEDLA_model = MEDLA(en1_chan_list = encoder_channel, en2_chan_list = encoder_channel2, en1_pool_list = encoder1_pool,
                    en1_ker_list = encoder1_kernel, en2_pool_list = encoder2_pool, en2_ker_list = encoder2_kernel)

    trainer = Trainer(dataloader_dict=dataloader_dict, model=MEDLA_model, lr= 0.001, num_epoch=1000)


    trainer.run_align()
    trainer.run_exp()
    trainer.run_finetune()