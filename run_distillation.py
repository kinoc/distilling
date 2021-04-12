import argparse
import json
import os
import pickle
import shutil

import numpy as np
import torch

from distiller import Distiller
from lm_seqs_dataset import LmSeqsDataset
from transformers import (
    GPTNeoConfig,
    GPTNeoForCausalLM, 
    GPT2Tokenizer,
)
from utils import init_gpu_params, logger, set_seed

student_config_class, student_model_class, _ = GPTNeoConfig, GPTNeoForCausalLM, GPT2Tokenizer
teacher_config_class, teacher_model_class, teacher_tokenizer_class = GPTNeoConfig, GPTNeoForCausalLM, GPT2Tokenizer
     
def freeze_pos_embeddings(student, args):
    student.transformer.wpe.weight.requires_grad = False
        
def train(args):
    # ARGS #
    init_gpu_params(args)
    set_seed(args)
    if args.is_master:
        if os.path.exists(args.dump_path):            
            if not args.force:
                logger.info(f"Overwrite flag --force is {args.force}")
                import time
                args.dump_path = f"{args.dump_path}_{str(time.time())}"
                os.makedirs(args.dump_path)
            else:
                shutil.rmtree(args.dump_path)


        if not os.path.exists(args.dump_path):
            os.makedirs(args.dump_path)
        logger.info(f"Experiment will be dumped and logged in {args.dump_path}")

        # SAVE PARAMS #
        logger.info(f"Param: {args}")
        with open(os.path.join(args.dump_path, "parameters.json"), "w",encoding = 'utf-8') as f:
            json.dump(vars(args), f, indent=4)
        #git_log(args.dump_path)

    # args.max_model_input_size = 1024
    # STUDENT #
    logger.info(f"Loading student config from {args.student_config}")
    stu_architecture_config = student_config_class.from_pretrained(args.student_config)
    stu_architecture_config.output_hidden_states = True

    if args.student_pretrained_weights is not None:
        logger.info(f"Loading pretrained weights from {args.student_pretrained_weights}")
        student = student_model_class.from_pretrained(args.student_pretrained_weights, config=stu_architecture_config)
    else:
        student = student_model_class(stu_architecture_config)

    if args.n_gpu > 0:
        student.to(f"cuda:{args.local_rank}")
    logger.info(f"Student loaded.")

    # TEACHER #
    teacher = teacher_model_class.from_pretrained(args.teacher_name, output_hidden_states=True)
    teacher.to('cuda')
    teacher.eval()
    if args.n_gpu > 0:
        teacher.to(f"cuda:{args.local_rank}")
    logger.info(f"Teacher loaded from {args.teacher_name}.")
    
    # pytorch DATASET #
    args.max_model_input_size = stu_architecture_config.max_position_embeddings
    logger.info(f"Loading data from {args.data_file}")
    with open(args.data_file, "rb") as fp:
        data = pickle.load(fp)
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.teacher_name)
    special_tok_ids = {"eos_token": tokenizer.eos_token,
                      "bos_token": tokenizer.bos_token,
                      "unk_token": tokenizer.unk_token}
    args.special_tok_ids = special_tok_ids
    
    train_lm_seq_dataset = LmSeqsDataset(params=args, data=data)
    logger.info(f"pytorch Dataset created.")

    # FREEZING #
    if args.freeze_pos_embs:
        freeze_pos_embeddings(student, args)

    # SANITY CHECKS #
    assert student.config.vocab_size == teacher.config.vocab_size
#     assert student.config.hidden_size == teacher.config.hidden_size
#     assert student.config.max_position_embeddings == teacher.config.max_position_embeddings

    # DISTILLER #
    torch.cuda.empty_cache()
    distiller = Distiller(
        params=args, dataset=train_lm_seq_dataset, token_probs=None, student=student, teacher=teacher
    )
    distiller.train()
    logger.info("Let's go get some drinks.")
    
def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--force", action="store_true", default=False, 
                        help="Overwrite dump_path if it already exists.")
    parser.add_argument("--dump_path", type=str, required=True, 
                        help="The output directory (log, checkpoints, parameters, etc.)")
    parser.add_argument("--data_file", type=str, required=True,
                        help="The binarized file (tokenized + tokens_to_ids) and grouped by sequence.")
    parser.add_argument("--student_config", type=str, required=True,
                        help="Path to the student configuration.")
    parser.add_argument("--student_pretrained_weights", default=None, type=str, 
                        help="Load student initialization checkpoint.")
    parser.add_argument("--teacher_name", default='EleutherAI/gpt-neo-1.3B',
                        help="The Teacher type DistilGPTNeo.",)
    parser.add_argument("--temperature", default=1.5, 
                        type=float, help="Temperature for the softmax temperature.")
    parser.add_argument("--alpha_ce", default=0.5, type=float, 
                        help="Linear weight for the distillation loss. Must be >=0.")
    parser.add_argument("--alpha_clm", default=0.5, type=float, help="Linear weight for the CLM loss. Must be >=0.")
    parser.add_argument("--alpha_mse", default=0.5, type=float, help="Linear weight of the MSE loss. Must be >=0.")
    parser.add_argument("--alpha_cos", default=0.5, type=float, help="Linear weight of the cosine embedding loss. Must be >=0.")
    parser.add_argument("--freeze_pos_embs", action="store_true",
                        help="Freeze positional embeddings during distillation")
    parser.add_argument("--n_epoch", type=int, default=3, help="Number of pass on the whole dataset.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (for each process).")
    parser.add_argument("--group_by_size", action="store_false",
            help="If true, group sequences that have similar length into the same batch. Default is true.",)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=50,
            help="Gradient accumulation for larger training batches.",)
    parser.add_argument("--warmup_prop", default=0.05, type=float, help="Linear warmup proportion.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")
    parser.add_argument("--initializer_range", default=0.02, type=float, help="Random initialization range.")
    parser.add_argument("--fp16",action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html",        )
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs in the node.")
    parser.add_argument("--local_rank", type=int, default=-1, help="Distributed training - Local rank")
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")

    parser.add_argument("--log_interval", type=int, default=500, help="Tensorboard logging interval.")
    parser.add_argument("--checkpoint_interval", type=int, default=1500, help="Checkpoint interval.")
    args = parser.parse_args()
    args.mlm = None
    args.alpha_mlm = 0.0
    args.restrict_ce_to_mask = False
    train(args)
        
if __name__ == "__main__":
    main()
