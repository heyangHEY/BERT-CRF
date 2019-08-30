python utils_zhijiang.py \
        --data_dir /ZJL/examples/ZJL/data/ \
        --train_review train/Train_reviews.csv \
        --train_result train/Train_labels.csv \
        --train_file train/train.csv \
        --test_review test/Test_reviews.csv \
        --test_file test/test.csv \
        --dev_file dev/dev.csv \
        --split_ratio 0.1

cuda=0
CUDA_VISIBLE_DEVICES=$cuda python run_zhijiang.py \
	--data_dir /ZJL/examples/ZJL/data/ \
	--model_type bert \
	--model_name_or_path bert-base-chinese \
	--task_name zhijiang \
	--output_dir /ZJL/examples/ZJL/result \
	--do_train \
    --evaluate_during_training \
	--per_gpu_train_batch_size 40 \
	--per_gpu_eval_batch_size 40 \
	--num_train_epochs 10.0 \
	--save_steps 10 \
	--logging_steps 10 \
	--learning_rate 5e-5 \
	--warmup_steps 73

# 	--evaluate_during_training \
#	--do_eval \
#	--eval_all_checkpoints \
#BERT_CHINESE_DIR=/ZJL/chinese_L-12_H-768_A-12/


#cuda=0
#CUDA_VISIBLE_DEVICES=$cuda python run_zhijiang.py \
#	--data_dir /ZJL/examples/ZJL/data/ \
#	--model_type bert \
#	--model_name_or_path bert-base-chinese \
#	--task_name zhijiang \
#	--output_dir /ZJL/examples/ZJL/result/checkpoint-640 \
#	--do_test \
#	--per_gpu_test_batch_size 8