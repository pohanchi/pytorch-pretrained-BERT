import json
import tqdm
import time
import pickle
def load_squad_dataset(dataset_path,cached=True):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    data = json.load(open(dataset_path,"r"))
    content = data["data"]
    data_list = list()
    print("generate data format!")
    start = time.time()
    for each_data in tqdm.tqdm(content):
        for qas_paragraph in each_data['paragraphs']:
            context = qas_paragraph['context']
            for qas in qas_paragraph['qas']:
                question=qas['question']
                ID =qas['id']
                try: 
                    if qas['is_impossible'] == False:
                        answer = qas['answers'][0]['text']
                    else:
                        answer = ""
                except:
                    print("this trap!")
                    answer = qas['answers'][0]['text']
                data_list+=[(context, question, answer, ID)]
    end = time.time()
    print("It took {} seconds on {} training data generation!".format(end-start,dataset_path))
    if cached:
        cached_prepro_data(data_list,dataset_path)
    return data_list
    

            
def cached_prepro_data(data_list,dataset_path):
    pickle.dump(data_list,open(dataset_path+".cached.p","wb"))



if __name__ == "__main__":
    load_squad_dataset("./squad-train-v1.1.json")
    load_squad_dataset("./squad-dev-v1.1.json")
    load_squad_dataset("./squad-train-v2.0.json")
    load_squad_dataset("./squad-dev-v2.0.json")
    load_squad_dataset("./coqa_to_squad-dev-v1.0.json")
    load_squad_dataset("./hotpot_to_squad-dev-v1.1.json")
    load_squad_dataset("./hotpot_to_squad-train-v1.1.json")
    load_squad_dataset("./coqa_to_squad-train-v1.0.json")
    load_squad_dataset("./quac_to_squad-train-v0.2.json")
    load_squad_dataset("./quac_to_squad-dev-v0.2.json")

    
