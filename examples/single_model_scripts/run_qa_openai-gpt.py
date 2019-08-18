# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
   WarmupLinearSchedule
   WarmupLinearSchedulet_val__spring2016\ -\ cloze_test_ALL_val.csv \
   WarmupLinearSchedule_test__spring2016\ -\ cloze_test_ALL_test.csv \
   WarmupLinearSchedule
   WarmupLinearSchedule
"""
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
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from utils import Regularization

from pytorch_transformers import (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,OpenAIGPTConfig,
                                     AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME, WarmupLinearSchedule)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def load_squad_dataset(dataset_path,cached=True,using_cache=False):
    """ Output a list of tuples(story, question, answer, ID) """
    print(dataset_path)
    if using_cache:
        data_list = pickle.load(open(dataset_path+".cached.p","rb"))
    else:
        data = json.load(open(dataset_path,"r"))
        content = data["data"]
        data_list = list()
        print("generate data format!")
        start = time.time()
        for each_data in tqdm(content):
            for qas_paragraph in each_data['paragraphs']:
                context = qas_paragraph['context']
                if len(context.split()) > 360:
                    context =  " ".join(context.split()[:360])
                for qas in qas_paragraph['qas']:
                    question=qas['question']
                    ID =qas['id']
                    try: 
                        if qas['is_impossible'] == False:
                            answer = qas['answers'][0]['text']
                        else:
                            answer = " "
                    except:
                        answer = qas['answers'][0]['text']
                    data_list+=[(context, question, answer, {ID})]
        end = time.time()
        print("It took {} seconds on {} training data generation!".format(end-start,dataset_path))
        if cached:
            cached_prepro_data(data_list,dataset_path)
    return data_list

def cached_prepro_data(data_list,dataset_path):
    pickle.dump(data_list,open(dataset_path+".cached.p","wb"))

def pre_process_datasets(encoded_datasets, input_len, cap_length,q_length,a_length,story_token, question_token,ans_token, end_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, :] = [story_token] + story[:story_length] + [question_token] + question[:que_length] +[end_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        for i, (story, question, ans,ID), in enumerate(dataset):
            context_question_answer = [story_token] + story[:cap_length] + [question_token] + question[:q_length] + [end_token] + [ans_token] + ans[:a_length]+[end_token]
            input_ids[i, :len(context_question_answer)] = context_question_answer
            lm_labels[i, :len(context_question_answer)] = context_question_answer
        all_inputs = (input_ids, lm_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

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
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
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
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ['_context_', '_question_', '_ans_','_eos_']
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    tokenizer.max_len = 1024
    tokenizer.add_tokens(special_tokens)
    # print("done!")
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTLMHeadModel(config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Load and encode the datasets
    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            # print("str ",obj)
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, set):
            return obj
        return list(tokenize_and_encode(o) for o in obj)
    logger.info("Encoding dataset...")
    train_dataset = load_squad_dataset(args.train_dataset,using_cache=args.using_cache)
    eval_dataset = load_squad_dataset(args.eval_dataset,using_cache=args.using_cache)
    datasets = (train_dataset, eval_dataset)
    encoded_datasets = tokenize_and_encode(datasets)
    # print("=============over!")
    # Compute the max input length for the Transformer
    max_length = model.config.n_positions//2 - 5
    q_length = 100
    a_length = 50
    
    input_length = max(len(story[:max_length]) + len(question[:q_length]) + len(ans[:a_length]) + 5  \
                           for dataset in encoded_datasets for story, question, ans, _ in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length,q_length,a_length ,*special_tokens_ids)
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]
    # print("=====done!!")
    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
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
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        if args.do_LLL != "":
            importance = args.importance
            Reg = Regularization(model=model,mode="EWC",dataset=train_dataloader)
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, lm_labels = batch
                losses = model(input_ids, labels=lm_labels)
                loss = losses[0] + importance * Reg.penalty(model)
                loss.backward()
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
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
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = OpenAIGPTLMHeadModel.from_pretrained(args.output_dir)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
        model.to(device)

    if args.do_eval:
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, lm_labels = batch
            with torch.no_grad():
                lm_loss, lm_logits = model(input_ids, lm_labels)

            lm_logits = lm_logits.detach().cpu().numpy()
            lm_labels = mc_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(lm_logits, lm_labels)

            eval_loss += lm_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == '__main__':
    main()
