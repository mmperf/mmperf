import os
import numpy as np
import tvm
from tvm import te, auto_scheduler

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a matmul with bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.


@auto_scheduler.register_workload
def matmul(M, N, K, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )

    return [A, B, C]


def autotune(M, N, K, target_name, dtype):
    ######################################################################
    # Create the search task
    # ^^^^^^^^^^^^^^^^^^^^^^
    # We then create a search task with N=L=M=1024 and dtype="float32"
    # If your machine supports avx instructions, you can
    #
    #   - replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
    #   - replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512
    target = tvm.target.Target(target_name)
    task = tvm.auto_scheduler.SearchTask(func=matmul, args=(M, N, K, dtype), target=target)
    # Inspect the computational graph
    print("Computational DAG:")
    print(task.compute_dag)

    ######################################################################
    # Next, we set parameters for the auto-scheduler.
    #
    # * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
    #   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
    #   good value for the search to converge. You can do more trials according to your time budget.
    # * In addition, we use :code:`RecordToFile` to dump measurement records into a file `matmul.json`.
    #   The measurement records can be used to query the history best, resume the search,
    #   and do more analyses later.
    # * see :any:`auto_scheduler.TuningOptions` for more parameters

    log_file = "matmul_{}x{}x{}.json".format(M, N, K)
    tune_option = None
    measure_ctx = None
    if target_name == "cuda":
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=1000,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )
    else:
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=1000,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=2,
        )

    ######################################################################
    # Run the search
    # ^^^^^^^^^^^^^^
    # Now we get all inputs ready. Pretty simple, isn't it?
    # We can kick off the search and let the auto-scheduler do its magic.
    # After some measurement trials, we can load the best schedule from the log
    # file and apply it.

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)

    if target_name == "cuda":
        del measure_ctx

    ######################################################################
    # We can lower the schedule to see the IR after auto-scheduling.
    # The auto-scheduler correctly performs optimizations including multi-level tiling,
    # layout transformation, parallelization, vectorization, unrolling, and operator fusion.

    print("Lowered TIR:")
    print(tvm.lower(sch, args, simple_mode=True))

    ######################################################################
    # Check correctness and evaluate performance
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # We build the binary and check its correctness and performance.
    func = tvm.build(sch, args, target, name="matmul")
    func.export_library("matmul_{}x{}x{}.so".format(M, N, K))
    if dtype == "float16":
        np_dtype = np.float16
    else:
        np_dtype = np.float32
    a_np = np.random.uniform(size=(M, K)).astype(np_dtype)
    b_np = np.random.uniform(size=(K, N)).astype(np_dtype)
    out_np = a_np.dot(b_np)

    ctx = None
    if target_name == "cuda":
        ctx = tvm.cuda()
    else:
        ctx = tvm.cpu()
    a_tvm = tvm.nd.array(a_np, device=ctx)
    b_tvm = tvm.nd.array(b_np, device=ctx)
    out_tvm = tvm.nd.empty(out_np.shape, dtype=dtype, device=ctx)
    func(a_tvm, b_tvm, out_tvm)

    # Check results
    np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=0.5)

    # Evaluate execution time.
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(a_tvm, b_tvm, out_tvm).results) * 1000)
    )

    ######################################################################
    # Using the record file
    # ^^^^^^^^^^^^^^^^^^^^^
    # During the search, all measurement records are dumped into the record
    # file "matmul.json". The measurement records can be used to re-apply search results,
    # resume the search, and perform other analyses.
    ######################################################################
    # Here is an example where we load the best schedule from a file,
    # and print the equivalent python schedule API. This can be used for
    # debugging and learning the behavior of the auto-scheduler.

    print("Equivalent python schedule:")
    print(task.print_best(log_file))

    if target_name == "cuda":
        print("CUDA source code:")
        print(task.print_best(log_file, print_mode="cuda"))

