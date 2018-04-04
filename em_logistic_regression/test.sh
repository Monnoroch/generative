echo "Running $0..."

set -e

STEPS=2
EXPERIMENT=test-exps/em-logistic-regression

python em_logistic_regression/train.py \
    --experiment_dir $EXPERIMENT \
    --batch_size 10 \
    --max_steps $STEPS

python em_logistic_regression/train.py \
    --experiment_dir $EXPERIMENT \
    --batch_size 10 \
    --max_steps $STEPS \
    --load_checkpoint $EXPERIMENT/model/checkpoint-$STEPS/data
