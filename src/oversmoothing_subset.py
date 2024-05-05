import numpy as np


import pickle
from datasets import load_dataset
import math


class OversmoothingSubsetFunction():
    
    
    def __init__(self, n, data, num_layers=2, sigma_value=0.1):
        self.n = n
        self.data = data
        self.num_layers = num_layers
        self.sigma_value = sigma_value
        
        if self.n <= 0:
            raise Exception("ERROR: Number of elements in ground set must be positive")
        
        
    def conv1d(self, vector, kernel):
        # Length of the input vector
        input_length = len(vector)
        # Length of the kernel
        kernel_length = len(kernel)
        # Padding size
        padding = kernel_length // 2
        # Pad the input vector
        padded_vector = np.pad(vector, (padding, padding), mode='constant')
        # Convolution result
        result = np.zeros(input_length)
        # Perform convolution
        for i in range(input_length):
            result[i] = np.sum(padded_vector[i:i+kernel_length] * kernel)
        return result
    
    
    def output_cnn(self,input_vector, kernel):
        if self.num_layers == 0:
            return input_vector
        curr_vector = input_vector
        for i in range(self.num_layers):
            hidden = self.conv1d(curr_vector, kernel)
            curr_vector = hidden
        output = hidden
        return np.array(output)
    
    
    def get_gaussian_kernel(self):
        filter_range = [-1, 0, 1]
        gaussian_filter = [1 / (self.sigma_value * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*self.sigma_value**2)) for x in filter_range]
        kernel = gaussian_filter / np.sum(gaussian_filter)
        return kernel
        
    def compute_instance_scores(self, data, centroids):
        instance_scores = []
        kernel = self.get_gaussian_kernel()
        
        clus0, clus1 = centroids
        smoothened_cluster0 = self.output_cnn(clus0, kernel)
        smoothened_cluster1 = self.output_cnn(clus1, kernel)
        
        for data_point in data:
            score = 0
            smoothened_vector = self.output_cnn(data_point, kernel)
            dist00 = np.sum((smoothened_vector - smoothened_cluster0) ** 2)
            dist01 = np.sum((smoothened_vector - smoothened_cluster1) ** 2)
            
            score+=np.abs(dist01-dist00)
            instance_scores.append(score)
            
        return instance_scores
            
        
    def maximize(self, budget, clusters):

        instance_scores = self.compute_instance_scores(self.data, clusters)
        instance_scores_sorted = sorted(enumerate(instance_scores), key=lambda x: x[1])
        return instance_scores_sorted[:budget]
    
    
    

dataset = load_dataset("christophsonntag/OLID")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

dataset_to_consider = train_dataset

# Read pkl file 
with open('/data/BADRI/ACAD/OML/Oversmoothing_In_Transformers/notebooks/embeddings/OLID-MPNET/OLID-train-768.pkl', 'rb') as f:
    vectors = pickle.load(f)
    
vectors = np.array(vectors)



    
c0 = 0 * vectors[0]
c1 = 0 * vectors[0]
count0 = 0
count1 = 0

for i in range(len(vectors)):
    item = dataset_to_consider[i]
    if item['subtask_a'] == 'NOT':
        c0 += vectors[i]
        count0 += 1
    else:
        c1 += vectors[i]
        count1 += 1
        
clus0 = c0 / count0
clus1 = c1 / count1

centroids = (clus0, clus1)

obj = OversmoothingSubsetFunction(n=len(vectors), data=vectors)
greedyList = obj.maximize(budget=int(0.75*len(vectors)), clusters=centroids)

print(greedyList)