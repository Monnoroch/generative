echo "Running $0..."

set -e

STEPS=2
EXPERIMENT=test-exps/gan-conv-nn
MNIST_DATA_DIR=mnist-data

python gan_conv_nn/train.py \
    --experiment_dir $EXPERIMENT \
    --max_steps $STEPS \
    --dataset_dir $MNIST_DATA_DIR \
    --batch_size 10 \
    --discriminator_steps 1 \
    --generator_steps 1 \
    --d_learning_rate 0.01 \
    --g_learning_rate 0.01 \
    --d_l2_reg 0.01 \
    --g_l2_reg 0.01 \
    --discriminator_features 10 \
    --discriminator_features 15 \
    --discriminator_filter_sizes 4 \
    --discriminator_filter_sizes 3 \
    --generator_features 10 \
    --generator_features 5 \
    --generator_filter_sizes 2 \
    --generator_filter_sizes 3 \
    --dropout 0.1 \
    --stride 2

python gan_conv_nn/train.py \
    --experiment_dir $EXPERIMENT \
    --load_checkpoint $EXPERIMENT/model/checkpoint-$STEPS/data \
    --max_steps 2 \
    --dataset_dir $MNIST_DATA_DIR \
    --batch_size 10 \
    --discriminator_steps 2 \
    --generator_steps 3 \
    --d_learning_rate 0.03 \
    --g_learning_rate 0.04 \
    --d_l2_reg 0.02 \
    --g_l2_reg 0.05 \
    --dropout 0.6

rm -rf $MNIST_DATA_DIR
