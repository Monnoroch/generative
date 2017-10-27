echo "Running $0..."

set +e

python logistic_regression/train.py --experiment_dir test-exps/logistic_regression \
    --l2_reg 0.01 --learning_rate 0.01 --input_mean 4 --input_stddev 1.5 --input_mean -2 --input_stddev 1 \
    --batch_size 3 --max_steps 2

python logistic_regression/train.py --experiment_dir test-exps/logistic_regression \
    --l2_reg 0.05 --learning_rate 0.05 --input_mean 6 --input_stddev 2.5 --input_mean -1 --input_stddev 2 \
    --batch_size 4 --max_steps 2 --load_checkpoint test-exps/logistic_regression/model/checkpoint-2/data
