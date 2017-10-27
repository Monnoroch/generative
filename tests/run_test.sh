rm -rf test-exps

export PYTHONPATH=.
$1
exitcode=$?

rm -rf test-exps

exit $exitcode
