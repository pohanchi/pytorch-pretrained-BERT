import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange
import json
import numpy as np
import torch
import time
import pickle
import pdb
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,BatchSampler,
                              TensorDataset)
from utils import Regularization
from pytorch_transformers import (GPT2LMHeadModel, GPT2Tokenizer,GPT2Config,
    AdamW,
    cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)
import json
import torch.nn.functional as F
import IPython

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)

def longest_length(model):
    max_length = model.config.n_positions - 100 - 4
    q_length = 50
    a_length = 50
    return max_length, q_length, a_length

def load_tool(model_name,special_tokens,device):
    load_dir = model_name
    tokenizer = GPT2Tokenizer.from_pretrained(load_dir)
    model = GPT2LMHeadModel.from_pretrained(load_dir)
    model.to(device)
    special_tokens_ids = list(
    tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    return load_dir, tokenizer, model, special_tokens_ids

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf'),device='cpu'):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim(
    # ) == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    # IPython.embed()
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        
        indices_to_remove = logits < torch.topk(logits,top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        # IPython.embed()
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        row_index = (sorted_indices_to_remove == 1).nonzero()[:,0]
        logits[row_index,indices_to_remove] = filter_value
    return logits


def sample_sequence(model,
                    context,
                    tokenizer,
                    num_samples=1,
                    temperature=1,
                    top_k=0,
                    top_p=0.9,
                    is_xlnet=False,
                    device='cpu',argmax=False):
    generated = context.clone().detach()
    generated = generated.unsqueeze(0)
    with torch.no_grad():
        end_token = "_eos_"
        end_word = tokenizer.convert_tokens_to_ids(end_token)
        next_token = "_none_"
        start_time = time.time()
        past = None
        count = 0
        while next_token != end_word:
            output = model(generated)
            
            next_token_logits = output[0][:, -1, :] / temperature
            
            if not argmax:
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p,device=device)
                next_tokens = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=num_samples,replacement=True)
                next_tokens_output=torch.mode(next_tokens, dim=-1, keepdim=True)
                next_token=next_tokens_output[0]
                
            else:
                next_token = torch.argmax(next_token_logits,keepdim=True,dim=-1)
                # IPython.embed()
                # pdb.set_trace()
            if count > 0:
                if next_token != end_word:
                    next_token_matrix = torch.cat((next_token_matrix,next_token),dim=1)
            else:
                if next_token != end_word:
                    next_token_matrix = next_token
            if next_token==end_word and count==0:
                return torch.tensor([[]])

            if count >=35:
                break

            generated = torch.cat((generated, next_token), dim=1)
            count += 1
        end_time = time.time()
        # print("It took {} seconds".format(end_time-start_time))

    return next_token_matrix

def load_squad_dataset(dataset_path, cached=True, using_cache=False):
    """ Output a list of tuples(story, question, answer, ID) """
    if using_cache:
        data_list = pickle.load(open(dataset_path + ".cached.p", "rb"))
    else:
        data = json.load(open(dataset_path, "r"))
        content = data["data"]
        data_list = list()
        start = time.time()
        for each_data in tqdm(content):
            for qas_paragraph in each_data['paragraphs']:
                context = qas_paragraph['context']
                for qas in qas_paragraph['qas']:
                    question = qas['question']
                    ID = qas['id']
                    try:
                        if qas['is_impossible'] == False:
                            answer = qas['answers'][0]['text']
                        else:
                            answer = ""
                    except:
                        answer = qas['answers'][0]['text']
                    data_list += [(context, question, answer, {ID})]
        end = time.time()
        print("It took {} seconds on {} training data generation!".format(
            end - start, dataset_path))
        if cached:
            cached_prepro_data(data_list, dataset_path)
    return data_list


def cached_prepro_data(data_list, dataset_path):
    pickle.dump(data_list, open(dataset_path + ".cached.p", "wb"))


def pre_process_datasets(datasets,input_len,story_token,question_token,ans_token,end_token,pad_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, :] = [story_token] + story[:story_length] + [question_token] + question[:que_length] +[end_token]
        answer = [ans_token] + ans[:a_length]+[end_token]
    """
    tensor_datasets = []
    a_length = 50
    for dataset in datasets:
        n_batch = len(dataset)
        ID_list = list()
        input_ids = np.zeros((n_batch,input_len))
        answer_span = np.zeros((n_batch,a_length))
        ID_indexes = np.zeros((n_batch, 1),dtype='int64')
        for i, (story, question, ans, ID), in enumerate(dataset):
            text=[story_token] + story + [question_token] + question + [ans_token]
            only_need_length = len(text)
            input_ids[i,:only_need_length] = text 
            input_ids[i,only_need_length:] = pad_token
            answer_span[i, :len(ans)] = ans
            answer_span[i,len(ans):] = pad_token
            ID_indexes[i, 0] = i
            ID_list.append(list(ID)[0])
            # IPython.embed()
            # pdb.set_trace()
        all_inputs = (input_ids, answer_span, ID_indexes)
        # print(len(input_list),len(answer_list),len(ID_indexes))
        # IPython.embed()
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        
    return tensor_datasets, ID_list


def main():
    """evaluate gpt-model fine-tune on qa dataset"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt2',
        help='pretrained model name')
    parser.add_argument("--using_cache", type=bool, default=False)
    parser.add_argument(
        "--importance", type=float, help="LifeLong Learning need its (Lambda)")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument("--do_eval",action="store_true")
    # parser.add_argument("--old_dataset", type=str, default="")
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    # parser.add_argument('--old_batch_size', type=int, default=1)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
    parser.add_argument("--no_cuda",action="store_true")
    parser.add_argument("--argmax",action="store_true")
    parser.add_argument("--sample",type=int,default=1)
    args = parser.parse_args()

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(args)
    set_seed(args.seed)
    device=args.device
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_eval:
        raise ValueError("At least `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    special_tokens = ['_context_', '_question_', '_ans_','_eos_','_pad_']
    load_dir, tokenizer, model, special_tokens_ids = load_tool(args.model_name,special_tokens,device)
    print(special_tokens_ids)

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            # print("str ",obj)
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, set):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    logger.info("Encoding dataset...")
    eval_dataset = load_squad_dataset(
        args.eval_dataset, using_cache=args.using_cache)
    datasets = (eval_dataset, )
    encoded_datasets = tokenize_and_encode(datasets)
    max_length, q_length, a_length = longest_length(model)

    input_length = max(len(story[:max_length]) + len(question[:q_length])  + 5  \
                            for dataset in encoded_datasets for story, question, ans, _ in dataset)
    input_length = min(input_length, model.config.n_positions-2) 

    # Load and encode the datasets

    # Prepare inputs tensors and dataloaders
    tensor_datasets, ID_list = pre_process_datasets(encoded_datasets, input_length,*special_tokens_ids)
    eval_data = TensorDataset(*tensor_datasets[0])
    eval_sampler = SequentialSampler(eval_data)
    eval_sampler=BatchSampler(eval_sampler,batch_size=args.eval_batch_size,drop_last=False)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,num_workers=8)

    if args.do_eval:
        model.eval()
        answer_dict = dict()
        compared_dict = dict()
        tqdm_bar = tqdm(eval_dataloader, desc="Evaluating")
        for step, data in enumerate(tqdm_bar):
            start_time = time.time()
            sentence, answer, ID_index = tuple(t.to(device) for t in data)
            sentence = sentence[sentence != special_tokens_ids[4]].long()
            answer = answer[answer != special_tokens_ids[4]].long()
            # print(answer)
            # pdb.set_trace()
            out = sample_sequence(
                model=model,
                context=sentence,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                is_xlnet=False,
                tokenizer=tokenizer,
                argmax=args.argmax,num_samples=args.sample)
            
            end_time = time.time()
            
            # print("It costs {} seconds for generate data!!".format(end_time-start_time))
            out_ = out[:, :].tolist()
            answer_ = tokenizer.decode(answer.tolist(),clean_up_tokenization_spaces=True)
            for i in range(len(out_)):
                text = tokenizer.decode(out_[i], clean_up_tokenization_spaces=True,skip_special_tokens=True)
                answer_dict[ID_list[ID_index[0][i]]] = text
                compared_dict[ID_list[ID_index[0][i]]] = (text,answer_)
                
            if step % 50 == 0:
                print("step:", step)
                print("  prediction: ",text)
                print("  groundtrut: ",answer_)
                

        with open(args.output_dir + "/predictions.json", "w") as outfile:
            json.dump(answer_dict, outfile)
        with open(args.output_dir + "/compared_answer.json","w") as outfile:
            json.dump(compared_dict, outfile)
    return

if __name__ == '__main__':
    main()
