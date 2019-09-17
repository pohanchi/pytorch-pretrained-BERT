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
import tensorboardX
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
from utils import Regularization
from pytorch_transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,OpenAIGPTConfig,AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)

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


def load_squad_dataset(path, cache=True,using_pickle=False):
    """generate data point for QA dataset"""
    """ a list of tuples(story, question, answer, ID)"""
    if using_pickle:
        data_list = pickle.load(open(path+".cached.p","rb"))
        return data_list
    else:
        data = json.load(open(path,"r"))
        data_list = list()
        start = time.time()
        for each_data in tqdm(data["data"]):
            for qas_paragraph in each_data['paragraphs']:
                context = qas_paragraph['context']
                for qas in qas_paragraph['qas']:
                    question = qas['question']
                    ID = qas['id']
                    try:
                        if qas['is_impossible'] == False:
                            answer = qas['answers'][0]['text']
                        else:
                            answer = " "
                    except:
                        answer = qas['answers'][0]['text']
                    data_list+=[(context, question, answer, {ID})]
        end = time.time()
        if cache:
            if not os.path.exists(path+".cached.p"):
                pickle.dump(data_list,open(dataset_path+".cached.p","wb"))
        return data_list

def pre_process_datasets(datasets,input_len,story_token,question_token,ans_token,end_token,pad_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, :] = [story_token] + story[:story_length] + [question_token] + question[:que_length] +[end_token]"""
    tensor_datasets = []
    for dataset in datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch,input_len),dtype=np.int64)
        lm_labels = np.full((n_batch,input_len),fill_value=-1,dtype=np.int64)
        for i , (story,question,ans, ID) in enumerate(dataset):
            cannot_calculate_loss = len([story_token] + story + [question_token] + question)
            text=[story_token] + story + [question_token] + question + [ans_token] + ans + [end_token]
            only_need_length = len(text)
            input_ids[i,:only_need_length] = text
            lm_labels[i,:only_need_length] = text
            input_ids[i,only_need_length:] = pad_token
            lm_labels[i,:cannot_calculate_loss] = -1
            
        all_inputs = (input_ids, lm_labels)
        tensor_datasets+= [tuple(torch.tensor(t) for t in all_inputs)]
    return tensor_datasets


def random_seed_setup(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    return

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    return device

def process_special_tokens():
    special_tokens = ['_context_', '_question_', '_ans_','_eos_','_pad_']
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    tokenizer.max_len = 1024
    tokenizer.add_tokens(special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    return  tokenizer,special_tokens_ids,special_tokens
    




def main():
    config = OpenAIGPTConfig.from_pretrained('openai-gpt')
    config.n_positions  = 1024
    config.n_ctx = 1024
    print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--using_cache", type=bool,default=False)
    parser.add_argument("--do_LLL", default="",type=str, help="which one ewc you want to run ? there have three method: EWC, SI, MAS, IMM")
    parser.add_argument("--importance",type=float,help="LifeLong Learning need its (Lambda)")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=5)
    parser.add_argument('--eval_batch_size', type=int, default=6)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training \
                        steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before\
                        performing a backward/update pass.")
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    args = parser.parse_args()
    print(args)

    random_seed_setup(args)
    device=setup_device()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer,special_tokens_ids,special_tokens =process_special_tokens()
    
    model = OpenAIGPTLMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    # model = nn.DataParallel(model)
    model.to(device)

    train_dataset = load_squad_dataset(args.train_dataset,using_pickle=args.using_cache)
    # print(train_dataset)
    # pdb.set_trace()
    dataset = (train_dataset,)
    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            # print(obj)
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, set):
            # print(obj)
            return obj
        # print(obj)
        return list(tokenize_and_encode(o) for o in obj)
        
    encoded_datasets = tokenize_and_encode(dataset)

    max_length, q_length, a_length = longest_length(model)

    input_length = max(len(story[:max_length]) + len(question[:q_length]) + len(ans[:a_length]) + 5  \
                            for dataset in encoded_datasets for story, question, ans, _ in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length,*special_tokens_ids)
    train_data = TensorDataset(*tensor_datasets[0])
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    # pdb.set_trace()
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    if args.do_train:
        writer = SummaryWriter("gpt-1-10epoch")
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        if args.do_LLL != "":
            importance = args.importance
            Reg = Regularization(model=model,mode="EWC",dataset=train_dataloader)
        step_step = 1
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels = batch
                losses = model(input_ids, labels=lm_labels)
                if args.do_LLL!="":
                    loss = losses[0] + importance * Reg.penalty(model)
                else:
                    loss = losses[0]
                loss.backward()
                optimizer.step()
                scheduler.step()
                writer.add_scalar('training_loss',loss,step_step)
                optimizer.zero_grad()
                tr_loss += loss.item()
                step_step += 1
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = OpenAIGPTLMHeadModel.from_pretrained(args.output_dir)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
        model.to(device)

if __name__ == "__main__":
    main()
