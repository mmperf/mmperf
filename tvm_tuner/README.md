## Run TVM’s Auto-scheduling

This code follows [TVM tutorial: Optimizing Operators with Auto-scheduling](https://tvm.apache.org/docs/tutorials/get_started/auto_scheduler_matmul_x86.html#optimizing-operators-with-auto-scheduling).

First, set the environment variable PYTHONPATH:

```
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

Then, run `tuner.py` to search for optimal schedules for each matmul input size. Specify the matrix sizes and target by setting `-m` and `-target` flags. For example, if you want to run search on Bert sizes with GPU, run the command:

```
python3 tuner.py -m ../benchmark_sizes/bert_large_matmul.txt -target cuda
```

After search is done, it automatically saves json and so files associate with each matrix size.