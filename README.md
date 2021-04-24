# distilling
Experiments with distilling large language models.

# alpha values 
- alpha_ce : Linear weight for KLDivLoss between student and teacher token classification outputs.
- alpha_clm : Linear weight for CrossEntropyLoss between stduent token classification outputs and the labels.
- alpha_mse : Linear weight for MSELoss between student and teacher token classification logits.
- alpha_cos : Linear weight for CosineEmbeddingLoss between student and teacher last layers hidden_states.(works only when the model hidden_size of student and teacher are same)

# STEPS
1. Use create_data.py (for small dataset) to create a pickle of input_ids list or use create_data_for_large_dataset.py (for large dataset) to create shards of npy files.
2. Use run_distillation.py to start distillation training. For training on large dataset set --train_on_large_dataset param.
