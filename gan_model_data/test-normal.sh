echo "Running $0..."

set +e

PYTHONPATH=. python gan_model/train.py --experiment_dir test-exps/gan-normal \
    --d_l2_reg 0.01 --input_mean 7 --input_stddev 1.5 --input_mean -1 --input_stddev 2 \
    --discriminator_steps 1 --generator_steps 1 --d_learning_rate 0.01 --g_learning_rate 0.01 --g_l2_reg 0.01 \
    --dropout 0.1 --generator_features 10 --discriminator_features 10 \
    --batch_size 10 --max_steps 2

PYTHONPATH=. python gan_model/train.py --experiment_dir test-exps/gan-normal \
    --d_l2_reg 0.04 --input_mean 8 --input_stddev 4.5 --input_mean -3 --input_stddev 1 \
    --discriminator_steps 2 --generator_steps 3 --d_learning_rate 0.03 --g_learning_rate 0.02 --g_l2_reg 0.05 \
    --batch_size 5 --max_steps 2 --load_checkpoint test-exps/gan-normal/model/checkpoint-2/data
