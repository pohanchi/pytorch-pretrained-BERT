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
        # indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
        #     dim=-1, index=sorted_indices.to(device), src=sorted_indices_to_remove.byte())
        logits[row_index,indices_to_remove] = filter_value
        # logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model,
                    context,
                    tokenizer,
                    num_samples=1,
                    temperature=1,
                    top_k=0,
                    top_p=0.9,
                    is_xlnet=False,
                    device='cpu'):
    # context = context.clone().detach()

    # context = context.unsqueeze(0).repeat(num_samples, 1)

    generated = context.clone().detach()
    with torch.no_grad():
        end_token = "_eos_"
        end_word = tokenizer.convert_tokens_to_ids(end_token)
        next_token = "_none_"
        start_time = time.time()
        past = None
        for i in range(20):
            # inputs = {'input_ids': generated,'past':past}
            # IPython.embed()
            output,past = model(generated)
            
            next_token_logits = output[:, -1, :] / temperature
            
            # start_m_time = time.time()

            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p,device=device)
            # end_m_time = time.time()
            # print("It took {} seconds".format(end_m_time-start_m_time))
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1), num_samples=1)
            # IPython.embed()
            # pdb.set_trace()
            # new_words = next_token
            if i > 0:
                next_token_matrix = torch.cat((next_token_matrix,next_token),dim=1)
            else:
                next_token_matrix = next_token
            generated = torch.cat((generated, next_token), dim=1)
        end_time = time.time()
        print("It took {} seconds".format(end_time-start_time))

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
        input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        answer_span = np.zeros((n_batch, a_length), dtype=np.int64)
        ID_list = list()
        ID_indexes = np.zeros((n_batch, 1),dtype='int64')
        for i, (story, question, ans, ID), in enumerate(dataset):
            cannot_calculate_loss = len([story_token] + story + [question_token] + question + [ans_token])
            text=[story_token] + story + [question_token] + question + [ans_token]
            only_need_length = len(text)
            input_ids[i,:only_need_length] = text
            input_ids[i,only_need_length:] = pad_token
            answer_span[i, :len(ans)] = ans
            ID_indexes[i, 0] = i
            ID_list.append(list(ID)[0])
        all_inputs = (input_ids, answer_span, ID_indexes)
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
    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set.")
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
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_batch_size', type=int, default=8)
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
    args = parser.parse_args()

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_eval:
        raise ValueError("At least `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    special_tokens = ['_context_', '_question_', '_ans_','_eos_','_pad_']
    load_dir = args.model_name
    tokenizer = GPT2Tokenizer.from_pretrained(load_dir)
    model = GPT2LMHeadModel.from_pretrained(load_dir)
    model.to(device)
    special_tokens_ids = list(
        tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
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

    input_length = max(len(story[:max_length]) + len(question[:q_length]) + len(ans[:a_length]) + 5  \
                            for dataset in encoded_datasets for story, question, ans, _ in dataset)
    input_length = min(input_length, model.config.n_positions) 

    # Load and encode the datasets

    # Prepare inputs tensors and dataloaders
    tensor_datasets, ID_list = pre_process_datasets(encoded_datasets, input_length,*special_tokens_ids)
    eval_data = TensorDataset(*tensor_datasets[0])
    eval_sampler = SequentialSampler(eval_data)
    eval_sampler=BatchSampler(eval_sampler,batch_size=args.eval_batch_size,drop_last=True)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler,num_workers=8)

    if args.do_eval:
        model.eval()
        answer_dict = dict()
        tqdm_bar = tqdm(eval_dataloader, desc="Evaluating")
        for step, data in enumerate(tqdm_bar):
            start_time = time.time()
            sentence, answer, ID_index = tuple(t.to(device) for t in data)
            
            # IPython.embed()
            # print(ID_index)
            out = sample_sequence(
                model=model,
                context=sentence[0],
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device,
                is_xlnet=False,
                tokenizer=tokenizer)
            # print(len(sentence[0]))
            end_time = time.time()
            # ID_index.data()
            
            print("It costs {} seconds for generate data!!".format(end_time-start_time))
            out_ = out[:, :].tolist()
            
            for i in range(len(out_)):
                text = tokenizer.decode(out_[i], clean_up_tokenization_spaces=True,skip_special_tokens=True)
                answer_dict[ID_list[ID_index[0][i]]] = text
            print(text)
        with open(args.output_dir + "/predictions.json", "w") as outfile:
            json.dumps(answer_dict, outfile)
    return

if __name__ == '__main__':
    main()
