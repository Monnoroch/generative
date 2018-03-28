echo "Running $0..."

set -e

python linear_regression/train.py --experiment_dir test-exps/linear_regression \
    --l2_reg 0.01 --learning_rate 0.01 --input_param1 0.8 --input_param2 1 --batch_size 10 --max_steps 2

python linear_regression/train.py --experiment_dir test-exps/linear_regression \
    --l2_reg 0.05 --learning_rate 0.05 --input_param1 1.8 --input_param2 2 --batch_size 5 --max_steps 2 \
    --load_checkpoint test-exps/linear_regression/model/checkpoint-2/data
