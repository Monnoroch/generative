rm -rf test-exps

PYTHONPATH=. python linear_regression/train.py --experiment_dir test-exps/linear_regression --l2_reg 0.01 --learning_rate 0.01 --input_param1 0.8 --input_param2 1 --batch_size 512 --max_steps 2
exitcode=$?

rm -rf test-exps

exit $exitcode
