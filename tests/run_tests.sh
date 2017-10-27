set +e

tests/run_test.sh "logistic_regression/test.sh"
tests/run_test.sh "gan_model/test-normal.sh"
tests/run_test.sh "gan_model/test-nn.sh"
