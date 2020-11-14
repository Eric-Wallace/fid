export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=16
python dense_retriever.py \
	--model_file checkpoint/retriever/single/nq/bert-base-encoder.cp.quantized \
	--ctx_file data/wikipedia_split/psgs_w100.tsv \
	--qa_file $1 \
	--save_or_load_index \
	--encoded_ctx_file small-indexes/single/nq/full/ \
        --index_factory_string PQ64 \
        --pca_dim 512 \
	--out_file retrieval_results.json \
	--n-docs 35 \
	--validation_workers 16 \
	--batch_size 64

python train_reader.py \
	--eval_top_docs 35 \
	--encoder_model_type t5 \
	--pretrained_model_cfg allenai/unifiedqa-t5-large-fid \
	--dev_file retrieval_results.json \
	--sequence_length 200 \
	--dev_batch_size 4 \
	--passages_per_question 35 \
	--passages_per_question_predict 35 \
	--model_file dpr_reader.9.934.quantized \
	--prediction_results_file reader_results.json

python postprocess_output.py $2
