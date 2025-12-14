import numpy as np
def load(a):
    a=str(a)
    loaded = np.load("endeavor.npz")
    dic={}
    dic["1"]=loaded["task1"]
    dic["2"]=loaded["task2"]
    dic["3"]=loaded["task3"]
    dic["4"]=loaded["task4"]
    dic["6a"]=loaded["task6a"]
    dic["6b"]=loaded["task6b"]
    dic["7a"]=loaded["task7a"]
    dic["7b"]=loaded["task7b"]
    return dic[a]
