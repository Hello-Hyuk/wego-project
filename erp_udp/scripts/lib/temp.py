import numpy as np
from wcwidth import list_versions

list = [x for x in range(10)]
list2 = [x+1 for x in range(10)]

list_np = np.array(list,dtype=np.float32)
list_np = np.reshape(list_np,(-1,1))

list_np2 = np.array(list2,dtype=np.float32)
list_np2 = np.reshape(list_np2,(-1,1))
# print(list_np)

print(f"list {list_np[:,0]}")
print(f"list {list_np2[:,0]}")

idx_range = np.where(list_np[:,0]<5.0)

list_np = np.delete(list_np,idx_range,axis=0)
list_np2 = np.delete(list_np2,idx_range,axis=0)
print(f"index range {np.where(list_np[:,0]<5.0)}")
        
print(list_np)     
print(list_np2)

