#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:41:33 2023

@author: malachyiii
"""

import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt


import torch

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

unique_people = []
people_vectors = []

features = torch.tensor([[1,2,3,4], [5,3,6,7], [8,16,5,9], [5,3,6,7], [5,3,6,7]], dtype = torch.float32).to(device)
target = torch.tensor([[1], [2], [2], [2], [3]]).to(device)

def loss_function(features, target, unique_people, people_vectors):
    losses = torch.empty((len(target)))
    
    for i in range(len(features)):
        
        if len(unique_people) == 0:
            unique_people.append(target[i])
            people_vectors = features[i].to(device).unsqueeze(0)
            losses[i] = torch.tensor([0])
            print("First Person Added")
        else:
            distances = torch.cdist(features[i].unsqueeze(0), people_vectors)

            if torch.min(distances) < 5 and unique_people[torch.argmin(distances)] == target[i]:
                print("Correct! (Positive)")
                losses[i] = torch.tensor([0])
            
            elif torch.min(distances) < 5 and target[i] not in unique_people:
                print("False Positive")
                losses[i] = torch.mean(distances)
            
            elif torch.min(distances) < 5 and unique_people[torch.argmin(distances)] != target[i]:
                print("Incorrect guess")
                loss = torch.cdist(features[i].unsqueeze(0), people_vectors[unique_people.index(target[i])].unsqueeze(0))
            
            elif torch.min(distances) > 5 and target[i] not in unique_people:
                print("Correct! (Negative)")
                people_vectors = torch.cat((people_vectors,features[i].unsqueeze(0)), dim = 0)
                unique_people.append(target[i])
                print(people_vectors)
                losses[i] = torch.tensor([0])
            
            elif torch.min(distances) > 5 and target[i] in unique_people:
                print("False Negative")
                loss = torch.cdist(features[i].unsqueeze(0), people_vectors[unique_people.index(target[i])].unsqueeze(0))
                
                losses[i] = loss
            


                   
            #if (target in unique_people) and (target == unique_people[]
    return losses, unique_people, people_vectors
    

loss, unique_people, people_vectors = loss_function(features, target, unique_people, people_vectors)
        
print(loss)
print(torch.mean(loss**2).requires_grad)

class MyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, target, unique_people, people_vectors):
        tensor = loss_function(features, target, unique_people, people_vectors)
        ctx.save_for_backward(tensor)
        return tensor

    @staticmethod
    def backward(ctx, grad_features):
        result, = ctx.saved_tensors
        return grad_features * result
        


MyLoss.apply(features, target, unique_people, people_vectors)