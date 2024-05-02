from submodlib import FacilityLocationFunction
import numpy as np
import pickle



with open('/data/BADRI/ACAD/OML/Oversmoothing_In_Transformers/data/embeddings/OLID-train-768.pkl', 'rb') as file:
    train_x = pickle.load(file)
  
  
train_x_temp = np.array(train_x)    
data_size = train_x_temp.shape[0]

print(data_size)
    
# objFL = FacilityLocationFunction(n=data_size, data=train_x_temp, mode="dense", metric="euclidean")
# from submodlib import DisparitySumFunction
# objDM = DisparitySumFunction(n=data_size, data=train_x_temp, mode="dense", metric="euclidean")

from submodlib import GraphCutFunction
objDM = GraphCutFunction(n=data_size, mode="dense", lambdaVal=0, data=train_x_temp, metric="euclidean")

print("Facility Location Application is done")

greedyList = objDM.maximize(budget=int(0.75*data_size),optimizer='LazyGreedy')



print(greedyList)
#save the greedyList in a file
with open('/data/BADRI/ACAD/OML/Oversmoothing_In_Transformers/data/olid_GC_LazyGreedy.pkl', 'wb') as file:
    pickle.dump(greedyList, file)