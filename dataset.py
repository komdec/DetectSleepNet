import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from utils import *
from tqdm import tqdm
# def get_subject_files(dataset_dir, sid, print_num=False):
#     """Get a list of files storing each subject data."""
#     num = 0
#     file_name = 'tr'+sid+'.npz'
#     file_path = os.path.join(dataset_dir, file_name)
#     # Get the subject files based on ID
#     return subject_files

def load_data(file_path):
    """Load data from subject files."""
    with np.load(file_path) as f:
        files = f.files
        x = f[files[0]][np.newaxis,:]
        for i in range(1,len(files)-1):
            x = np.append(x, f[files[i]][np.newaxis,:],0)
        y = f['y'].astype(np.int32)
        # Reshape the data to match the input of the model - conv2d
        # Casting
        x = x.astype(np.float32)
    return x, y


def load_sleep_data(data_path='preprocessed'):
    data = []
    labels = []
    for file in os.listdir(data_path):
        if not file.endswith('.npz'):
            continue
        path = os.path.join(data_path, file)
        loaded = np.load(path)
        data.append(loaded['x'])
        labels.append(loaded['y'])
    return data, labels


class BatchSleepDataset_bak(Dataset):
    def __init__(self, dataset_dir, ids, minibatch_size=20):
        self.dataset_dir = dataset_dir
        # self.ids = ids
        self.data_len = []
        self.files_path = []
        for i in range(len(ids)):
            file_name = 'tr' + ids[i] + '.npz'
            file_path = os.path.join(self.dataset_dir, file_name)
            self.files_path.append(file_path)
            with np.load(file_path) as f:
                self.data_len.append(len(f['y']))
        self.minibatch_size = minibatch_size
        self.ids = ids
        self.reshuffle()

    def reshuffle(self):
        self.shifts = []
        self.cur_lens = []
        self.total_len = 0
        for subj_data_len in self.data_len:
            max_skip = 5 * self.minibatch_size + subj_data_len % self.minibatch_size
            cur_skip = random.randint(0, max_skip)
            self.shifts.append(cur_skip)
            cur_len = (subj_data_len - cur_skip) // self.minibatch_size
            self.cur_lens.append(cur_len)
            self.total_len += cur_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        subj_idx = len(self.ids) - 1
        for idx in range(len(self.ids)):
            cur_len = self.cur_lens[idx]
            if index >= cur_len:
                index -= cur_len
            else:
                subj_idx = idx
                break
        start_idx = self.shifts[subj_idx] + index * self.minibatch_size
        data = np.load(self.files_path[subj_idx])
        data_channel = data.files[:-1]
        item_data = []
        for channel_i in data_channel:
            item_data.append(data[channel_i][start_idx:start_idx + self.minibatch_size])
        item_data = np.array(item_data)
        # try:
        #     assert item_data.shape[1]==self.minibatch_size
        # except:
        #     breakpoint()
        item_labels = data[data.files[-1]][start_idx:start_idx + self.minibatch_size]  
        try:
            result_dataset = torch.tensor(item_data, dtype=torch.float), \
               torch.tensor(item_labels, dtype=torch.long)   
        except:
            breakpoint()
        return result_dataset


                
class SleepDataset(Dataset):
    def __init__(self, dataset_dir, ids, dataset_name):
        self.trains_x , self.trains_y =[], []
        self.ids = ids
        subject_files = glob.glob(os.path.join(dataset_dir, "*.npz"))
        subject_files.sort()
        if dataset_name =='phy2018':
            assert len(subject_files) == 994
        elif dataset_name =='shhs1':
            assert len(subject_files) == 5793
        for i in tqdm(self.ids,desc='Test data to RAM',leave=False):
            file_name = subject_files[i]
            train_x, train_y= load_data(file_name)
            self.trains_x.append(train_x)
            self.trains_y.append(train_y)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return torch.tensor(self.trains_x[index], dtype=torch.float),\
            torch.tensor(self.trains_y[index], dtype=torch.long)

class SplitSleepDataset(Dataset):
    def __init__(self, dataset_dir, ids, data_memory,fake=False,split=2):
        self.dataset_dir = dataset_dir
        self.ids = ids
        self.memory = data_memory
        if self.memory:
            self.trains_x , self.trains_y =[], []
            for i in tqdm(range(len(self.ids)),desc='val data to RAM',leave=False):
                if fake:
                    file_name = self.ids[i] + '.npz'
                else:
                    file_name = 'tr' + self.ids[i] + '.npz'
                file_path = os.path.join(self.dataset_dir, file_name)
                train_x, train_y= load_data(file_path)
                train_x = np.array_split(train_x,split,1)
                train_y = np.array_split(train_y,split)
                for train_idx in range(len(train_x)):
                    self.trains_x.append(train_x[train_idx])
                    self.trains_y.append(train_y[train_idx])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.memory:
            return torch.tensor(self.trains_x[index], dtype=torch.float),\
                torch.tensor(self.trains_y[index], dtype=torch.long)
        else:
            file_name = 'tr' + self.ids[index] + '.npz'
            file_path = os.path.join(self.dataset_dir, file_name)
            train_x, train_y= load_data(file_path)
            return torch.tensor(train_x, dtype=torch.float),\
                    torch.tensor(train_y, dtype=torch.long)
            
# class SleepDataset_memory(Dataset):
#     def __init__(self, dataset_dir, ids):
#         self.dataset_dir = dataset_dir
#         self.ids = ids
#         self.trains_x , self.trains_y =[], []
#         for i in range(len(self.ids)):
#             file_name = 'tr' + self.ids[i] + '.npz'
#             file_path = os.path.join(self.dataset_dir, file_name)
#             train_x, train_y= load_data(file_path)
#             self.trains_x.append(train_x)
#             self.trains_y.append(train_y)
#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, index):
#         return torch.tensor(self.trains_x[index], dtype=torch.float),\
#                 torch.tensor(self.trains_y[index], dtype=torch.long)

def sleep_stage2w(array):
    assert len(array.shape) == 1
    new_array = np.ones_like(array)
    w_idx = np.where(array==0)[0]
    new_array[w_idx]=0
    return new_array

def sleep_stage2w_r(array):
    assert len(array.shape) == 1
    new_array = np.ones_like(array)
    w_idx = np.where(array==0)[0]
    r_idx = np.where(array==4)[0]
    new_array[w_idx]=0
    new_array[r_idx]=2
    return new_array



class BatchSleepDataset_name(Dataset):
    def __init__(self, subject_files, ids, minibatch_size=20,data_memory=False,fake=False,stage=None):
        self.subject_files = subject_files
        self.memory = data_memory
        # self.ids = ids
        self.data_len = []
        self.datas_list = []
        self.labels_list = []
        self.files_path = []
        for i in tqdm(ids,desc='train data to RAM',leave=False):
            file_path = subject_files[i]
            with np.load(file_path) as f:
                if self.memory:
                    data_channel = f.files[:-1]
                    item_data = []
                    for channel_i in data_channel:
                        item_data.append(f[channel_i])
                    item_label = f['y']
                    if stage == 'w':                        
                        item_label = sleep_stage2w(item_label)
                    elif stage == 'w_r':                        
                        item_label = sleep_stage2w_r(item_label)
                    elif stage == None:
                        pass
                    else:
                        breakpoint()                        
                    item_data = np.array(item_data)
                    self.datas_list.append(item_data)
                    self.labels_list.append(item_label)
                else:
                    self.files_path.append(file_path)
                self.data_len.append(len(f['y']))
                
        self.minibatch_size = minibatch_size
        self.ids = ids
        self.reshuffle()

    def reshuffle(self):
        self.shifts = []
        self.cur_lens = []
        self.total_len = 0
        if self.minibatch_size==999: #999: auto_w batch max_minibatch=80, min_minibatch=20
            max_minibatch=40
            min_minibatch=20
            self.data999 = []
            self.label999 = []
            for data_idx in range(len(self.data_len)):
                subj_data_new = []
                subj_label_new = []
                subj_data = self.datas_list[data_idx][0]
                subj_label = self.labels_list[data_idx]
                sub_w_idx = np.where(subj_label==0)[0]
                sub_w_skip = np.where(np.diff(sub_w_idx)!=1)[0]
                for data_idx_j in range(len(sub_w_skip)-1):
                    if data_idx_j  == 0:
                        subjj_label = subj_label[:sub_w_idx[sub_w_skip[data_idx_j+1]]+1]
                        subjj_data = subj_data[:sub_w_idx[sub_w_skip[data_idx_j+1]]+1]
                    else:
                        subjj_label = subj_label[sub_w_idx[sub_w_skip[data_idx_j-1]+1]:sub_w_idx[sub_w_skip[data_idx_j+1]]+1]
                        subjj_data = subj_data[sub_w_idx[sub_w_skip[data_idx_j-1]+1]:sub_w_idx[sub_w_skip[data_idx_j+1]]+1]

                    # subj_label[sub_w_idx[sub_w_skip[data_idx]]+1:sub_w_idx[sub_w_skip[data_idx+1]]+1]
                
                    while len(subjj_label) < min_minibatch and data_idx_j<len(sub_w_skip)-1:
                        
                        try:
                            subjj_label = np.append(subjj_label,subj_label[sub_w_idx[sub_w_skip[data_idx_j]]+1:sub_w_idx[sub_w_skip[data_idx_j+1]]+1])
                            subjj_data = np.append(subjj_data,subj_data[sub_w_idx[sub_w_skip[data_idx_j]]+1:sub_w_idx[sub_w_skip[data_idx_j+1]]+1],0)
                        except:
                            breakpoint()
                        data_idx_j+=1

                    if len(subjj_label) > max_minibatch:
                        subjj_label = subjj_label[:max_minibatch]
                        subjj_data = subjj_data[:max_minibatch,:]
                    
                    if len(subjj_label) < max_minibatch:
                        subjj_label = np.append(subjj_label,np.zeros((max_minibatch-len(subjj_label))))                       
                        subjj_data = np.append(subjj_data,np.zeros((max_minibatch-len(subjj_data),subjj_data.shape[-1])),0)
                    self.total_len+=1
                    subj_data_new.append(subjj_data)
                    subj_label_new.append(subjj_label)
                try:
                    subj_data_new = np.concatenate(subj_data_new,0)
                    subj_label_new = np.concatenate(subj_label_new,0)
                except:
                    break
                    breakpoint()
                cur_len = len(subj_label_new)//max_minibatch
                self.cur_lens.append(cur_len)
                self.data999.append(subj_data_new)
                self.label999.append(subj_label_new)                   
            self.max_minibatch = max_minibatch



        else:
            for subj_data_len in self.data_len:
                max_skip = 5 * self.minibatch_size + subj_data_len % self.minibatch_size
                cur_skip = random.randint(0, max_skip)
                self.shifts.append(cur_skip)
                cur_len = (subj_data_len - cur_skip) // self.minibatch_size
                self.cur_lens.append(cur_len)
                self.total_len += cur_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        subj_idx = len(self.ids) - 1
        for idx in range(len(self.ids)):
            cur_len = self.cur_lens[idx]
            if index >= cur_len:
                index -= cur_len
            else:
                subj_idx = idx
                break
        
        if self.memory:
            if self.minibatch_size==999:
                item_data = self.data999[subj_idx][index*self.max_minibatch:(index+1)*self.max_minibatch,:][np.newaxis,:,:]
                item_labels = self.label999[subj_idx][index*self.max_minibatch:(index+1)*self.max_minibatch][np.newaxis,:]
            else:
                start_idx = self.shifts[subj_idx] + index * self.minibatch_size
                item_data = self.datas_list[subj_idx][:,start_idx:start_idx + self.minibatch_size,:]
                item_labels = self.labels_list[subj_idx][start_idx:start_idx + self.minibatch_size]
        else:
            start_idx = self.shifts[subj_idx] + index * self.minibatch_size
            data = np.load(self.files_path[subj_idx])
            data_channel = data.files[:-1]
            item_data = []
            for channel_i in data_channel:
                item_data.append(data[channel_i][start_idx:start_idx + self.minibatch_size])
            item_data = np.array(item_data)
            # try:
            #     assert item_data.shape[1]==self.minibatch_size
            # except:
            #     breakpoint()
            item_labels = data[data.files[-1]][start_idx:start_idx + self.minibatch_size]  
        try:
            result_dataset = torch.tensor(item_data, dtype=torch.float), \
            torch.tensor(item_labels, dtype=torch.long)   
        except:
            breakpoint()
        return result_dataset    

# def load_split_sleep_dataset(data_path='preprocessed', minibatch_size=20):
#     data, labels = load_sleep_data(data_path)
#     train_size = round(len(data) - 8)
#     data_train, labels_train = data[:train_size], labels[:train_size]
#     data_test, labels_test = data[train_size:], labels[train_size:]
#     return BatchedSleepDataset(data_train, labels_train, minibatch_size), SleepDataset(data_test, labels_test)

last_epoch_dict = {'0':[0],
                   '1':[1],
                   '2':[2],
                   '3':[3],
                   '4':[4],
                   'all':[0,1,2,3,4]}

class BatchSleepDataset_5classes(Dataset):
    def __init__(self, dataset_dir, ids, fake=False,diff=True,out_lastclass='all',split=False):
        assert out_lastclass in ['0','1','2','3','4','all']
        out_lastclass = last_epoch_dict[out_lastclass]
        self.dataset_dir = dataset_dir
        self.data_len = 0
        self.split = split
        self.datas_list =  [[] for _ in range(5)]
        self.labels_list =  [[] for _ in range(5)]
        self.files_path = []
        for i in tqdm(range(len(ids)),desc='train data to RAM',leave=False):
            if fake:
                file_name = ids[i] + '.npz'
            else:
                file_name = 'tr' + ids[i] + '.npz'
            file_path = os.path.join(self.dataset_dir, file_name)            
            with np.load(file_path) as f:
                f_file = f.files
                assert len(f_file)==2
                item_data = f[f_file[0]][:,np.newaxis,:]
                item_label = f[f_file[1]]
                # self.data_len+=len(item_label)

                ## 5classes

                item_data_lastc_idx = [[] for _ in range(5)]
                item_label_paded = np.insert(item_label,0,0)
                if diff:
                    item_idx_diff = np.where(np.diff(item_label_paded)!=0)[0]
                for item_idx in range(len(item_label_paded)-1): 
                    if diff and item_idx not in item_idx_diff and item_idx not in out_lastclass:
                        continue
                    item_data_lastc_idx[item_label_paded[item_idx]].append(item_idx)
                    
                for c_i in range(5):
                    self.datas_list[c_i].append(item_data[item_data_lastc_idx[c_i]])
                    self.labels_list[c_i].append(item_label[item_data_lastc_idx[c_i]])
        for c_i in range(5):
            self.datas_list[c_i] = np.concatenate(self.datas_list[c_i])
            self.labels_list[c_i] = np.concatenate(self.labels_list[c_i])  
            if self.data_len<len(self.labels_list[c_i]):
                self.data_len=len(self.labels_list[c_i])
        if out_lastclass == 'all' and not split:
            self.datas_list = np.concatenate(self.datas_list)   
            self.labels_list = np.concatenate(self.labels_list)
            self.data_len=len(self.labels_list)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.split:
            item_data = []
            item_label = []
            for c_i in range(5):
                item_idx = index % len(self.datas_list[c_i])
                item_data.append(self.datas_list[c_i][item_idx])
                item_label.append(self.labels_list[c_i][item_idx][np.newaxis])
            item_data = np.concatenate(item_data)

            item_label = np.concatenate(item_label)
                
        else:
            item_data = torch.cat(self.datas_list[index])
            item_label = self.labels_list[index]
        result_dataset = torch.tensor(item_data, dtype=torch.float), torch.tensor(item_label, dtype=torch.long)   
        return result_dataset    

def get_subject_files(dataset, files, sid, print_num=False):
    # breakpoint()
    """Get a list of files storing each subject data."""
    num = 0
    reg_exp = f"[fake|SC|ST][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
    # Get the subject files based on ID
    subject_files = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_files.append(f)
            num+=1
    if print_num:
        return subject_files,num
    else:
        return subject_files

def get_subject_idx(dataset, files, sid, print_num=False):
    # breakpoint()
    """Get a list of files storing each subject data."""
    reg_exp = f"[fake|SC|ST][4|7]{str(sid).zfill(2)}[a-zA-Z0-9]+\.npz$"
    # Get the subject files based on ID
    subject_idx = []
    for i, f in enumerate(files):
        pattern = re.compile(reg_exp)
        if pattern.search(f):
            subject_idx.append(i)
    return subject_idx

    
    
def Get_corresponding_idx(dataset_name, subject_files, train_sids, valid_sids, test_sids):
    train_idx = []
    for sid in train_sids:
        train_idx.append(get_subject_idx(
            dataset=dataset_name,
            files=subject_files,
            sid=sid,
        ))
    if len(train_idx)>0:
        train_idx = np.hstack(train_idx)
    
    valid_idx= []
    for sid in valid_sids:
        valid_idx.append(get_subject_idx(
            dataset=dataset_name,
            files=subject_files,
            sid=sid,
        ))

    if len(valid_idx)>0:
        valid_idx = np.hstack(valid_idx)


    test_idx = []
    for sid in test_sids:
        test_idx.append(get_subject_idx(
            dataset=dataset_name,
            files=subject_files,
            sid=sid))
    if len(test_idx)>0:
        test_idx = np.hstack(test_idx)
    return train_idx, valid_idx, test_idx
    
def Get_corresponding_files(dataset_dir, train_sids, valid_sids, test_sids,fake=False):
    train_files = []
    for sid in train_sids:
        if fake:
            file_name = sid + '.npz'
        else:
            file_name = 'tr' + sid + '.npz'
        # file_name = 'tr' + sid + '.npz'
        train_files.append(os.path.join(dataset_dir, file_name))       
    train_x, train_y, _ = load_data(train_files)
    train_data = train_x, train_y
    
    valid_files = []
    for sid in valid_sids:
        if fake:
            file_name = sid + '.npz'
        else:
            file_name = 'tr' + sid + '.npz'        
        # file_name = 'tr' + sid + '.npz'
        valid_files.append(os.path.join(dataset_dir, file_name))
    val_x, val_y, _ = load_data(valid_files)
    valid_data = val_x, val_y


    test_files = []
    for sid in test_sids:
        if fake:
            file_name = sid + '.npz'
        else:
            file_name = 'tr' + sid + '.npz'        
        # file_name = 'tr' + sid + '.npz'
        test_files.append(os.path.join(dataset_dir, file_name))
    test_x, test_y, _ = load_data(valid_files)
    test_data = test_x, test_y
    return train_data, valid_data, test_data

class BatchSleepDataset_emg(Dataset):
    def __init__(self, dataset_dir, emg_dir, ids, minibatch_size=20,data_memory=False):
        self.dataset_dir = dataset_dir
        self.memory = data_memory
        # self.ids = ids
        self.data_len = []
        self.datas_list = []
        self.labels_list = []
        self.files_path = []
        for i in tqdm(range(len(ids)),desc='train data to RAM',leave=False):
            file_name = 'tr' + ids[i] + '.npz'
            emg_name = 'tr' + ids[i] + '.npy'
            file_path = os.path.join(self.dataset_dir, file_name) 
            emg_path = os.path.join(emg_dir, emg_name)          
            with np.load(file_path) as f:
                emg_f =  np.load(emg_path)[np.newaxis,:]
                if self.memory:
                    data_channel = f.files[:-1]
                    item_data = []
                    for channel_i in data_channel:
                        item_data.append(f[channel_i])
                    item_label = f['y']
                    item_data = np.array(item_data)
                    item_data = np.concatenate((item_data,emg_f),2)
                    self.datas_list.append(item_data)
                    self.labels_list.append(item_label)
                else:
                    self.files_path.append(file_path)
                self.data_len.append(len(f['y']))
        self.minibatch_size = minibatch_size
        self.ids = ids
        self.reshuffle()

    def reshuffle(self):
        self.shifts = []
        self.cur_lens = []
        self.total_len = 0
        for subj_data_len in self.data_len:
            max_skip = 5 * self.minibatch_size + subj_data_len % self.minibatch_size
            cur_skip = random.randint(0, max_skip)
            self.shifts.append(cur_skip)
            cur_len = (subj_data_len - cur_skip) // self.minibatch_size
            self.cur_lens.append(cur_len)
            self.total_len += cur_len

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        subj_idx = len(self.ids) - 1
        for idx in range(len(self.ids)):
            cur_len = self.cur_lens[idx]
            if index >= cur_len:
                index -= cur_len
            else:
                subj_idx = idx
                break
        start_idx = self.shifts[subj_idx] + index * self.minibatch_size
        if self.memory:
            item_data = self.datas_list[subj_idx][:,start_idx:start_idx + self.minibatch_size,:]
            item_labels = self.labels_list[subj_idx][start_idx:start_idx + self.minibatch_size]
        else:
            data = np.load(self.files_path[subj_idx])
            data_channel = data.files[:-1]
            item_data = []
            for channel_i in data_channel:
                item_data.append(data[channel_i][start_idx:start_idx + self.minibatch_size])
            item_data = np.array(item_data)
            # try:
            #     assert item_data.shape[1]==self.minibatch_size
            # except:
            #     breakpoint()
            item_labels = data[data.files[-1]][start_idx:start_idx + self.minibatch_size]  
        try:
            result_dataset = torch.tensor(item_data, dtype=torch.float), \
            torch.tensor(item_labels, dtype=torch.long)   
        except:
            breakpoint()
        return result_dataset    

class SleepDataset_emg(Dataset):
    def __init__(self, dataset_dir, emg_dir, ids, data_memory):
        self.dataset_dir = dataset_dir
        self.ids = ids
        self.memory = data_memory
        if self.memory:
            self.trains_x , self.trains_y =[], []
            for i in tqdm(range(len(self.ids)),desc='val data to RAM',leave=False):
                file_name = 'tr' + self.ids[i] + '.npz'
                emg_name = 'tr' + self.ids[i] + '.npy'
                file_path = os.path.join(self.dataset_dir, file_name)
                emg_path = os.path.join(emg_dir, emg_name)   
                emg_f = np.load(emg_path)[np.newaxis,:]      
                train_x, train_y= load_data(file_path)
                train_x = np.concatenate((train_x,emg_f),2)
                self.trains_x.append(train_x)
                self.trains_y.append(train_y)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.memory:
            return torch.tensor(self.trains_x[index], dtype=torch.float),\
                torch.tensor(self.trains_y[index], dtype=torch.long)
        else:
            file_name = 'tr' + self.ids[index] + '.npz'
            file_path = os.path.join(self.dataset_dir, file_name)
            train_x, train_y= load_data(file_path)
            return torch.tensor(train_x, dtype=torch.float),\
                    torch.tensor(train_y, dtype=torch.long)