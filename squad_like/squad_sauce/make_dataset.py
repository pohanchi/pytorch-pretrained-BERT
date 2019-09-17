import pickle as pe 
import numpy as np 
import json 
import IPython
import pdb 


def define_dataset(first_dataset, second_dataset):
    first_list =pe.load(open(first_dataset,"rb"))
    second_list=pe.load(open(second_dataset,"rb"))

    print("1.1 version: length=",len(first_list))
    print("2.0 version: length=",len(second_list))
    #IPython.embed()
    first_list += second_list[:8760] #add 10% length of 1.1 data from squad 2.0 dataset
    second_list = second_list[8760:] # throw 10% length of 1.1 data to 1.1

    pe.dump(first_list,open("squad-1.1-mixed10.p","wb")) 
    pe.dump(second_list,open("squad-2.0-minus10.p","wb"))

    return

if __name__ == "__main__":
    define_dataset("../squad-train-v1.1.json.cached.p","../squad-train-v2.0.json.cached.p")