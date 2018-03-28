echo "Running $0..."

set -e

NUM_STEPS=2

python gan_model_data/train.py --experiment_dir test-exps/gan-normal \
    --d_l2_reg 0.001 --input_mean 7 --input_stddev 1.5 --input_mean -1 --input_stddev 2 \
    --discriminator_steps 2 --generator_steps 2 --d_learning_rate 0.01 --g_learning_rate 0.01 --g_l2_reg 0.0001 \
    --dropout 0.5 --generator_features 10 --discriminator_features 10 --nn_generator \
    --batch_size 10 --max_steps $NUM_STEPS

CHECKPOINT=test-exps/gan-normal/model/checkpoint-$NUM_STEPS/data

python gan_model_data/generate_real.py --load_checkpoint $CHECKPOINT \
    --samples 10 --input_mean 7 --input_stddev 1.5 \
    | python gan_model_data/discriminate.py --load_checkpoint $CHECKPOINT

python gan_model_data/generate_fake.py --load_checkpoint $CHECKPOINT \
    --samples 10 \
    | python gan_model_data/discriminate.py --load_checkpoint $CHECKPOINT
