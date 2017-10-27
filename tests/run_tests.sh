set +e

tests/run_test.sh "logistic_regression/test.sh"
tests/run_test.sh "gan_model_data/test-normal.sh"
tests/run_test.sh "gan_model_data/test-nn.sh"
