dataset='/your/data/path/'
#directory structure
#N4News
#--imgs
#--news

GPU=4

#training
CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset $dataset  --output checkpoints --expert_config '444' --beta 0.1  --batch_size 192

#testing
CUDA_VISIBLE_DEVICES=$GPU python eval.py --dataset $dataset  --output checkpoints --expert_config '444' --beta 0.1