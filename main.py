#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import math
import numpy as np


# In[2]:


# # Import Kaggle data
# import kaggle
# kagggle competitions download tabular-playground-series-mar-2021


# In[3]:


# Add data to memory for discovery
train,test = pd.read_csv('./train.csv',index_col="id"),pd.read_csv('./test.csv',index_col="id")


# In[15]:


# Get data structure
train.info()


# In[16]:


# Peek at data examples
train.head()


# In[17]:


# Get data structure
test.info()


# In[18]:


# Peek at data examples
test.head()


# In[6]:


# Get number of unique categories
cal_cols = train.select_dtypes('object')
cal_cols_keys = cal_cols.keys()
# cal_cols_vals = np.array((cal_cols.values))

#Build sample item
cat_stats = np.array([train[c].nunique() for c in cal_cols])
median_cat = math.ceil(np.median(cat_stats))
target_cat_index = int(np.where(cat_stats == median_cat)[0])
target_cat_name = cal_cols_keys[target_cat_index]

print("Categroy with median total unique values:\n",target_cat_name,": ",train[target_cat_name].nunique(),"unique values")


# In[7]:


# Index categories
categoricals_to_ix = {}

for c in train.select_dtypes('object'):
    categoricals_to_ix[c] = {cat:i for i,cat in enumerate(train[c].unique())}

#Confirm with sample indexing
sample_cat = categoricals_to_ix[target_cat_name]
print(target_cat_name,"indices:\n",sample_cat)


# In[8]:


# Test embeddings
test_embeds = {}

for cix_key in categoricals_to_ix.keys():
    cix = categoricals_to_ix[cix_key]
    bag_size = len(cix)
    embed_size = math.isqrt(bag_size) #hyperparameter to optimize later
    test_embeds[cix_key] = nn.Embedding(bag_size,embed_size)

#Confirm with sample embedding
target_cat_keys = list(sample_cat.keys())
target_cat_vals = np.array(list(sample_cat.values()))
target_cat_vals_median = math.ceil(np.median(target_cat_vals))
target_cat_keys_sample = target_cat_keys[target_cat_vals_median]
print("Sanple",target_cat_name,"category:\n",target_cat_keys_sample)
    
testor = torch.tensor([categoricals_to_ix[target_cat_name][target_cat_keys_sample]], dtype=torch.long)
cat_keys_sample_hash = test_embeds[target_cat_name](testor)
print("Sanple category embedding:\n",cat_keys_sample_hash)


# In[9]:


# Define embeddings content (dictionary and translations)
#function to build embeddings library
#function assumes only categorical features are fed in
def BuildEmbedlibrary(features):
    categoricals_to_ix,embeds = {},{}
    
    #Build category index
    for c in features:
        categoricals_to_ix[c] = {cat:i for i,cat in enumerate(train[c].unique())}
    
    #Create embeddings for each feature
    for cix_key in categoricals_to_ix.keys():
        cix = categoricals_to_ix[cix_key]
        bag_size = len(cix)
        embed_size = math.isqrt(bag_size) #hyperparameter to optimize later
        embeds[cix_key] = nn.Embedding(bag_size,embed_size)
    
#         print(embeds[cix_key])
    return categoricals_to_ix,embeds

#function that gets respective embedding for given feature value
def CatToSubvector(category,feature_name,categorical_library,embedding_library,verbose=False):
#    if verbose:
#         print(verbose)
#         for info in [categorical_library]:
# #         for info in [category,feature_name,categorical_library,embedding_library]:
#             print(info,"\n")
#        print("category: {}; feature_name: {}; categorical_library: {}; embedding_library: {}".format(
#            category,
#            feature_name,
#            categorical_library,
#            embedding_library
#        ))
#     [print(valu,"\n") for valu in [category,feature_name,categorical_library,embedding_library]]
    raw_idx = torch.tensor([categorical_library[feature_name][category]], dtype=torch.long)
#     if (verbose): print("embedding_library[feature_name]:",embedding_library[feature_name],"\n","raw_idx:",raw_idx)
    emb = embedding_library[feature_name](raw_idx)
#     if (verbose): print("embedding_library[feature_name](raw_idx):",emb,"\n")
#     subvector = torch.Tensor(emb)
#     if (verbose): print("subvector:",subvector)
    
#     return subvector
    return emb


# In[10]:


# # Confirm function structure
# test_cat_dict,test_embeddings = BuildEmbedlibrary(train.select_dtypes('object'))
# test_subvec = CatToSubvector(target_cat_keys_sample,target_cat_name,test_cat_dict,test_embeddings)

# # print("Sanple category embedding:\n",test_subvec)


# In[19]:


# Define data

#define dataset
class ClaimsDataset(Dataset):
    def __init__(self, csv_path='./train.csv',labelled=False):
        """
        Args:
            raw_data (dataframe): Dataframe with raw featurres and labels.
        """
#         self.categoricals,self.embeddings = {},{}
        raw_data = pd.read_csv(csv_path)
        
#         self.l = len(raw_data)
        self.target = raw_data.pop('target') if labelled else torch.tensor([], dtype=torch.long)
        categorical_features = raw_data.select_dtypes(include='object')
        noncategorical_features = raw_data.select_dtypes(exclude='object')
        
        self.categoricals,self.embeddings = BuildEmbedlibrary(categorical_features)
        
        subvects = DataFrame()
#         for row in categorical_features.values:
#             subsubvects = []
#             for v,c in zip(row,categorical_features.columns):
#                 subsubvects.append(CatToSubvector(v,c,self.categoricals,self.embeddings))
# #             [print("subsubvect size:",ssv.size()) for ssv in subsubvects]
#             subvects.append(torch.cat(subsubvects,1))

        subsubvects = []
        for col in categorical_features.columns:
            cat_col = (categorical_features[col]).tolist()
#             print("cat_col:",cat_col,"\n")
#             cat_emb = cat_col
#             cat_emb = cat_col.apply(lambda x: CatToSubvector(x,col,self.categoricals,self.embeddings,True))
#             if (i%2):# == 0 and i!=0):
#                 print("i:",i)
#                 cat_emb = cat_col.apply(lambda x: CatToSubvector(x,col,self.categoricals,self.embeddings,True))
#                 print("cat_emb:",cat_emb,"\n")
#             else:
#                 cat_emb = cat_col.apply(lambda x: CatToSubvector(x,col,self.categoricals,self.embeddings,False))
            cat_embs = CatToSubvector(cat_col[0],col,self.categoricals,self.embeddings,True)
            for x in cat_col[1:]:
                new_emb = CatToSubvector(x,col,self.categoricals,self.embeddings,True)
                cat_embs = torch.cat((cat_embs,new_emb))
#             cat_emb = [CatToSubvector(x,col,self.categoricals,self.embeddings) for x in cat_col]
#             print(col,"cat_embs.shape:",cat_embs.shape,"\n")
#             cat_emb = torch.cat(tuple(cat_emb))
#             print("type(cat_emb[-10]):",type(cat_emb[-10]),"\n")
#             print("cat_emb[-10]:",cat_emb[-10],"\n")
            subsubvects.append(cat_embs)
            
#         print("tuple(subsubvects):",tuple(subsubvects),"\n")
        subvects = torch.cat(tuple(subsubvects),1)
#         print("subvects[1]:",subvects[1],"\n")
#         print("subvects.shape:",subvects.shape,"\n")


        
        # copy the data 
        df_min_max_scaled = noncategorical_features.copy() 

        # apply normalization techniques by Column 1 
        for col in df_min_max_scaled.columns:
            df_min_max_scaled[col] = (df_min_max_scaled[col] - df_min_max_scaled[col].min()) / (df_min_max_scaled[col].max() - df_min_max_scaled[col].min())

#         noncat_mean = noncategorical_features.mean(1)
#         noncat_std = noncategorical_features.std(1)
        
        
        self.data = torch.cat((
            subvects,
            torch.tensor(noncategorical_features.values)
        ),1)
                
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
            
        return x,y
    
    def __len__(self):
        return len(self.vectors)


# In[20]:


def LoadData(batch_size_order=2):
    bs = 2*batch_size_order
    train_set,test_set = ClaimsDataset('./train.csv',labelled=True),ClaimsDataset('./test.csv')
    train_loader,test_loader = DataLoader(claimset,batch_size=bs,shuffle=True)
    
    return trainloader,testloader


# In[21]:


trainloader,testloader = LoadData()
# print("cd.categoricals:",cd.categoricals,"\n","cd.embeddings:",cd.embeddings,"\n","cd.vectors.shape:",cd.vectors.shape)
x,y = trainloader[0]
print("x:",x)
print("y:",y)


# In[ ]:




