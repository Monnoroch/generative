echo "Running $0..."

set -e

STEPS=2
EXPERIMENT=test-exps/gan-nn
MNIST_DATA_DIR=mnist-data

python gan_deep_nn/train.py \
    --experiment_dir $EXPERIMENT \
    --dataset_dir $MNIST_DATA_DIR \
    --d_l2_reg 0.01 \
    --discriminator_steps 1 \
    --generator_steps 1 \
    --d_learning_rate 0.01 \
    --g_learning_rate 0.01 \
    --g_l2_reg 0.01 \
    --dropout 0.1 \
    --generator_features 10 \
    --generator_features 5 \
    --discriminator_features 10 \
    --discriminator_features 15 \
    --batch_size 10 \
    --max_steps $STEPS

python gan_deep_nn/train.py \
    --experiment_dir $EXPERIMENT \
    --dataset_dir $MNIST_DATA_DIR \
    --d_l2_reg 0.02 \
    --discriminator_steps 2 \
    --generator_steps 3 \
    --d_learning_rate 0.03 \
    --g_learning_rate 0.04 \
    --g_l2_reg 0.05 \
    --dropout 0.6 \
    --batch_size 10 \
    --max_steps 2 \
    --load_checkpoint $EXPERIMENT/model/checkpoint-$STEPS/data

rm -rf $MNIST_DATA_DIR
