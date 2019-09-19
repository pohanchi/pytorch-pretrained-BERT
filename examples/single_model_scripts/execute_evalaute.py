import subprocess
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str,help="you want to evaluate!!!")
    parser.add_argument("--output_dir", type=str,help="the target directory!!!")
    parser.add_argument("--version",type=str,help="1.1 or 2.0")
    parser.add_argument("--model_file",type=str, help="the model path")
    args = parser.parse_args()
    mode = [ "argmax","normal","sample10"]
    for m in mode:
        if m == "argmax":
            subprocess.call(["python", "./evaluate_qa_gpt.py","--model_name",args.model_file,"--eval_dataset",args.data_file,"--do_eval","--output_dir","./eval_2/"+args.output_dir+"/test"+args.version+"/argmax","--eval_batch_size","1", "--argmax"])
            subprocess.call(["python","../utils_squad_evaluate.py",args.data_file,"./eval_2/"+args.output_dir+"/test"+args.version+"/argmax"+"/predictions.json","-o","./eval_2/"+args.output_dir+"/test"+args.version+"/argmax"+"/eval.json"])
        if m == "normal":
            subprocess.call(["python", "./evaluate_qa_gpt.py","--model_name",args.model_file,"--eval_dataset",args.data_file,"--do_eval","--output_dir","./eval_2/"+args.output_dir+"/test"+args.version+"/normal","--eval_batch_size","1", "--sample","1",])
            subprocess.call(["python","../utils_squad_evaluate.py",args.data_file,"./eval_2/"+args.output_dir+"/test"+args.version+"/normal"+"/predictions.json","-o","./eval_2/"+args.output_dir+"/test"+args.version+"/normal"+"/eval.json"])
        if m == "sample10":
            subprocess.call(["python", "./evaluate_qa_gpt.py","--model_name",args.model_file,"--eval_dataset",args.data_file,"--do_eval","--output_dir","./eval_2/"+args.output_dir+"/test"+args.version+"/sample10","--eval_batch_size","1", "--sample","10",])
            subprocess.call(["python","../utils_squad_evaluate.py",args.data_file,"./eval_2/"+args.output_dir+"/test"+args.version+"/sample10"+"/predictions.json","-o","./eval_2/"+args.output_dir+"/test"+args.version+"/sample10"+"/eval.json"])

