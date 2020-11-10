# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7,8,9
python -m torch.distributed.launch --nproc_per_node 6 train_reader.py \
--seed 42 \
--learning_rate 1e-4  \
--eval_step 1000 \
--eval_top_docs 100  \
--encoder_model_type t5 \
--pretrained_model_cfg allenai/unifiedqa-t5-large-fid \
--gold_passages_src data/data/gold_passages_info/nq_train.json \
--gold_passages_src_dev data/data/gold_passages_info/nq_dev.json \
--train_file data/data/retriever_results/nq/single/train.json \
--dev_file data/data/retriever_results/nq/single/dev.json \
--warmup_steps 0 \
--sequence_length 250 \
--batch_size 1 \
--gradient_accumulation_steps 8 \
--passages_per_question 100 \
--num_train_epochs 100000 \
--dev_batch_size 2 \
--passages_per_question_predict 100 \
--log_batch_step 20 \
--output_dir myoutput_dir \
--fp16
