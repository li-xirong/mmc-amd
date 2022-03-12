### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import copy

def remove_normal_temp(A_path_list):
    A_path_list_new = copy.deepcopy(A_path_list)
    for item in A_path_list:
        filename = os.path.split(item)[-1]
        if filename[2]=='h':
            A_path_list_new.remove(item)
    return A_path_list_new

def remove_unmatched_data(A_path_list, B_path_list_with_unmatched):
    A_path_list = remove_normal_temp(A_path_list)
    A_filename_list = [os.path.split(item)[-1] for item in A_path_list]
    
    B_filename_list_with_unmatched = [os.path.split(item)[-1] for item in B_path_list_with_unmatched]
    B_root = [os.path.split(item)[0] for item in B_path_list_with_unmatched]
    
    B_path_list = []
    for i, item in enumerate(B_filename_list_with_unmatched):
        if item+'.npy' in A_filename_list:
            B_path_list.append(os.path.join(B_root[i], item))
 
    return sorted(A_path_list), sorted(B_path_list)
    

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.camlabel = opt.camlabel

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        A_paths = sorted(make_dataset(self.dir_A))

        if opt.phase == 'train':
            ### input B (real images)
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            B_paths = sorted(make_dataset(self.dir_B))
            self.A_paths, self.B_paths = remove_unmatched_data(A_paths, B_paths)
            assert len(self.A_paths) == len(self.B_paths)
        elif opt.phase == 'test':
            self.A_paths = A_paths

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def weighted_sigmoid(self, arr, w=1):
        return 1. / (1 + np.exp(-arr * w))      

    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]
        if not self.camlabel:              
            A = Image.open(A_path)        
            params = get_params(self.opt, A.size)
            if self.opt.label_nc == 0:
                transform_A = get_transform(self.opt, params)
                A_tensor = transform_A(A.convert('RGB'))
            else:
                transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                A_tensor = transform_A(A) * 255.0
        else:
            A = np.load(A_path)
            A = self.weighted_sigmoid(A, 0.1) * 255
            A = A[1:]
            A = np.transpose(A, (1,2,0))
            A = Image.fromarray(A.astype('uint8')).convert('RGB')
            params = get_params(self.opt, A.size)
            
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A)


        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)      
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))                            

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
