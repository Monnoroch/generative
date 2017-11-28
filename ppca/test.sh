echo "Running $0..."

set +e

STEPS=2
EXP_NAME=test-exps/ppca

python ppca/train.py \
    --experiment_dir $EXP_NAME \
    --batch_size 8 \
    --learning_rate 0.01 \
    --latent_space_size 1 \
    --input_mean 0.5 \
    --input_mean 1. \
    --input_stddev 1. \
    --input_stddev 0.1 \
    --max_steps $STEPS

python ppca/train.py \
    --experiment_dir $EXP_NAME \
    --batch_size 9 \
    --learning_rate 0.02 \
    --input_mean 0.6 \
    --input_mean 1.1 \
    --input_stddev 0.9 \
    --input_stddev 0.15 \
    --max_steps 2 \
    --load_checkpoint $EXP_NAME/model/checkpoint-$STEPS/data
