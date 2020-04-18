import torch
import torch.nn as nn
import numpy as np

class ConditionalSimNet(nn.Module):
    def __init__(self, embeddingnet, n_conditions, embedding_size, learnedmask=True, prein=False):
        """ embeddingnet: The network that projects the inputs into an embedding of embedding_size
            n_conditions: Integer defining number of different similarity notions
            embedding_size: Number of dimensions of the embedding output from the embeddingnet
            learnedmask: Boolean indicating whether masks are learned or fixed
            prein: Boolean indicating whether masks are initialized in equally sized disjoint 
                sections or random otherwise"""
        super(ConditionalSimNet, self).__init__()

        self.learnedmask  = learnedmask
        self.embeddingnet = embeddingnet

        print('=================================================================')
        print('ConditionalSimNet : ', self.learnedmask, prein)        

        # create the mask
        if learnedmask:
            if prein:
                # define masks 
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                # initialize masks
                mask_array = np.zeros([n_conditions, embedding_size])
                mask_array.fill(0.1)
                mask_len = int(embedding_size / n_conditions)
                for i in range(n_conditions):
                    mask_array[i, i*mask_len:(i+1)*mask_len] = 1
                # no gradients for the masks
                self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=True)
            else:
                print('\tdefine masks with gradients')
                # define masks with gradients
                self.masks = torch.nn.Embedding(n_conditions, embedding_size)
                print('\tself.masks : ' , self.masks) # Embedding(4, 64)
                # initialize weights
                self.masks.weight.data.normal_(0.9, 0.7) # 0.1, 0.005
        else:
            # define masks 
            self.masks = torch.nn.Embedding(n_conditions, embedding_size)
            # initialize masks
            mask_array = np.zeros([n_conditions, embedding_size])
            mask_len = int(embedding_size / n_conditions)
            for i in range(n_conditions):
                mask_array[i, i*mask_len:(i+1)*mask_len] = 1
            # no gradients for the masks
            self.masks.weight = torch.nn.Parameter(torch.Tensor(mask_array), requires_grad=False)

    def forward(self, x, c):
      
        #print('x : ', x.shape) # [256, 3, 112, 112]
        embedded_x = self.embeddingnet(x) # [256, 64]
        #print('embedded_x : ', embedded_x.shape)
        self.mask = self.masks(c) # [256, 64]
        #print('mask : ', self.mask.shape)
        
        if self.learnedmask:
            self.mask = torch.nn.functional.relu(self.mask)
            #print('mask : ', self.mask.shape) # [256, 64]
        masked_embedding = embedded_x * self.mask
        #print('masked_embedding : ', masked_embedding.shape) # torch.Size([256, 64])
        
        return masked_embedding, self.mask.norm(1), embedded_x.norm(2), masked_embedding.norm(2)


