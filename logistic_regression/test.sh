echo "Running $0..."

set -e

STEPS=2
EXPERIMENT=test-exps/logistic-regression

python logistic_regression/train.py \
    --experiment_dir $EXPERIMENT \
    --input_mean 4 \
    --input_stddev 1.5 \
    --input_mean -2 \
    --input_stddev 1 \
    --l2_reg 0.01 \
    --learning_rate 0.01 \
    --batch_size 3 \
    --max_steps $STEPS

python logistic_regression/train.py \
    --experiment_dir $EXPERIMENT \
    --load_checkpoint $EXPERIMENT/model/checkpoint-$STEPS/data \
    --input_mean 6 \
    --input_stddev 2.5 \
    --input_mean -1 \
    --input_stddev 2 \
    --l2_reg 0.05 \
    --learning_rate 0.05 \
    --batch_size 4 \
    --max_steps $STEPS
