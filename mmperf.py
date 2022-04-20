#!/usr/bin/env python3

import sys
import argparse
import os
import os.path
import platform
import time
import subprocess
import shutil
import re
import collections
import signal
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import GPUtil
import csv
import json
import torch
import triton

plt.style.use('ggplot')

BAR_WIDTH = 0.15
BAR_COLORS = {'mkl': 'cornflowerblue',
              'accelerate': 'lightgray',
              'mlir': 'olivedrab',
              'mlircuda': 'green',
              'openblas': 'wheat',
              'blis': 'mediumspringgreen',
              'blasfeo': 'sandybrown',
              'cublas': 'chocolate',
              'halide': 'gold',
              'ruy': 'violet',
              'tvm': 'indigo',
              'tvmcuda': 'darkslateblue',
              'naive': 'black',
              'nodai': 'orangered',
              'ireevmvx': 'thistle',
              'ireedylib': 'aqua',
              'ireecuda': 'deeppink',
              'mlir-sandbox': 'mediumseagreen',
              'triton': 'purple',
              'nodai-mlir-sandbox': 'red',
              'nodai-iree': 'red',
              'nodai-shark-cuda': 'red'}
BENCHMARK_ENV = os.environ.copy()
BENCHMARK_ENV.update({
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "HL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "TVM_NUM_THREADS": "1",
})

def path_expand(s):
    return Path(s).expanduser().resolve()

def add_arguments(parser):
    parser.add_argument('bins', type=path_expand,
                        help='Path where the test binaries are')
    parser.add_argument('results', type=path_expand,
                        help='Result directory')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel jobs for running the benchmarks')
    # Flags for mlir-sandbox and nodai-mlir-sandbox
    parser.add_argument('-sandbox', action='store_true',
                        help='Whether to run matmul in iree-llvm-sandbox')
    parser.add_argument('-num_iters', dest='num_iters', type=int, default=100,
                        help='Number of iterations to run each matmul')
    parser.add_argument('-benchmark_path', dest='benchmark_path',
                        help='Path to matmul size list for mlir-sandbox search')
    parser.add_argument('-nodai_configs', dest='nodai_configs',
                        help='Path to load config files generated for nodai-mlir-sandbox')
    parser.add_argument('-sandbox_configs', dest='sandbox_configs',
                        help='Path to load config files generated for mlir-sanxbox')
    # Flags to enable triton
    parser.add_argument('-triton', action='store_true',
                        help='Whether to run matmul in triton')
    parser.add_argument('-dtype', default='fp32',
                        help='Data precision for triton benchmark')

def make_result_dir(base_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    result_dir = (base_dir / timestamp).resolve()
    os.makedirs(result_dir)
    latest_symlink = os.path.join(base_dir, 'latest')
    print("Latest symlink path is: ", latest_symlink)
    print("Latest results path is: ", result_dir)
    # Remove old latest link
    if os.path.isdir(latest_symlink):
        os.unlink(latest_symlink)
    cwd = os.getcwd()
    os.chdir(base_dir)
    os.symlink(timestamp, 'latest')
    os.chdir(cwd)
    return result_dir

def write_system_info(output_dir, cpuinfo_dir):
    pfm = platform.system()

    if pfm == "Linux":
        # linux
        print("Linux System Detected.. looking for /proc/cpuinfo")
        shutil.copyfile(Path("/proc/cpuinfo"), output_dir / "cpuinfo")

        cpu_pattern = re.compile('cpu[0-9]+')
        cpudirs = [x for x in Path("/sys/devices/system/cpu/").iterdir() if cpu_pattern.match(x.name)]

        with open(output_dir / 'scaling_governor', 'w') as f:
            for cpu in cpudirs:
                sc_gov = (cpu / 'cpufreq' / 'scaling_governor')
                if sc_gov.is_file():
                    f.write(cpu.name + ": " + sc_gov.read_text())
                else:
                    f.write(cpu.name + ": not available")

        with open(output_dir / 'core_frequencies', 'w') as f:
            for cpu in cpudirs:
                sc_freq = (cpu / 'cpufreq' / 'scaling_cur_freq')
                if sc_freq.is_file():
                    f.write(cpu.name + ": " + sc_freq.read_text())
                else:
                    f.write(cpu.name + ": not available")

    elif pfm == "Darwin":
        # OSX
        print("OSX System Detected")
    else:
        print("Unidentified system")

    with open(output_dir / 'arch-info', 'w') as fh:
        proc = subprocess.run([cpuinfo_dir / "bin" / "cpu-info"],
                              capture_output=True, text=True, check=True)
        fh.write(proc.stdout)
        proc = subprocess.run([cpuinfo_dir / "bin" / "isa-info"],
                              capture_output=True, text=True, check=True)
        fh.write(proc.stdout)
        proc = subprocess.run([cpuinfo_dir / "bin" / "cache-info"],
                              capture_output=True, text=True, check=True)
        fh.write(proc.stdout)
    
    # Obtain GPU information if available
    try:
        GPUs = GPUtil.getGPUs()
        # TODO: investigate why GPUs gets set to empty list in some cases
        if len(GPUs) > 0:
            with open(output_dir / 'gpu-info', 'w') as fg:
                gpu_name = GPUs[0].name
                fg.write(gpu_name)
    except:
        pass

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        # If the floor value of GFLOPS is 0 print its float value
        if int(height) == 0:
            plt.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%.3f' % float(height), fontsize=5, ha='center', va='bottom')
        else:
            plt.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%d' % int(height), fontsize=5, ha='center', va='bottom')

_result_dir = None
_env = None

def _do_single_permutation(i, path, msize):
    try:
        cmd = f'{path} --benchmark_format=csv > result_{path.name}.csv'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, check=True, cwd=_result_dir)
        output = "result_" + path.name + ".csv"
    
        # parse the CPU benchmark results, the elapse time is shown as 'Duration(nsec)'
        with open(os.path.join(_result_dir, output), 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            runtime = 0
            for line in csv_reader:
                if (line[0].startswith('BM_Matmul')):
                    duration = float(line[3])
                    time_unit = {'ns': 0, 'us': 1, 'ms': 2, 's': 3}
                    factor = [1e9, 1e6, 1e3, 1]
                    runtime = duration / factor[time_unit[line[4]]]
            if runtime == 0.0:
                speed = 0.0
            else:
                mat_size = [float(m) for m in msize.split('x')]
                mnk_prod = np.prod(mat_size)
                speed = 2.0 * mnk_prod / runtime / 1e9
    
        gflops_path = _result_dir / (path.name + '_perf.out')
        with open(gflops_path, 'w') as f:
            f.write(str(speed) + " GFLOPS")
            f.close()

        return i, speed, runtime
    except:
        return i, False, 0.0

def _gpu_nsys_permutation(i, path, msize, perm_name, warm_up_runs=5):
    try:
        cmd = f'sudo /usr/local/cuda/bin/nsys profile -t nvtx,cuda -o {_result_dir}/qdrep/report_{path.name}.qdrep -f true {path}'
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, check=True, cwd=_result_dir)
        cmd = f'sudo /usr/local/cuda/bin/nsys stats -f csv --report gputrace {_result_dir}/qdrep/report_{path.name}.qdrep > result_{path.name}.csv'
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, check=True, cwd=_result_dir)
        nsys_output = "result_" + path.name + ".csv"

        if perm_name == 'cublas':
            name_start = ['volta_sgemm', 'void gemm', 'void gemv']
        else:
            name_start = ['matmul']
        
        # parse the nsys results, the elapse time is shown as 'Duration(nsec)'
        with open(os.path.join(_result_dir, nsys_output), 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            duration = 0
            cnt = 0
            for line in csv_reader:
                if len(line) == 0: continue
                if line[-1].startswith(tuple(name_start)):
                    cnt += 1
                    if cnt > warm_up_runs:  # warp up runs are excluded
                        duration += float(line[1])

            runtime = duration / (cnt - warm_up_runs) / 1e9
            if runtime == 0.0:
                speed = 0.0
            else:
                mat_size = [float(m) for m in msize.split('x')]
                mnk_prod = np.prod(mat_size)
                speed = 2.0 * mnk_prod / runtime / 1e9

        gflops_path = _result_dir / (path.name + '_perf.out')
        with open(gflops_path, 'w') as f:
            f.write(str(speed) + " GFLOPS")
            f.close()

        return i, speed, runtime
    except:
        return i, False, 0.0

def sandbox_perf(file_path, num_iters, use_configs=False):
    try:
        if use_configs == True:
            cmd = f'python -m python.examples.matmul.iree_sandbox_matmul -config_path {file_path} -n_iters {num_iters}'
        else:
            cmd = f'python -m python.examples.matmul.iree_sandbox_matmul -matrix_path {file_path} -n_iters {num_iters}'
        dst_f = './external/iree-llvm-sandbox'
        result = subprocess.run(cmd, shell=True, check=True, cwd=dst_f)
    except subprocess.TimeoutExpired:
        print("\033[31m" + "FAILED" + "\033[m")
        print("  -> Execution timed out")
        return False
    if result.returncode != 0:
        print("\033[31m" + "FAILED" + "\033[m")
        print(f"  -> Returned error code {result.returncode}")
        return False

    if use_configs == True:
        output_file = 'nodai_sandbox_matmul_results.json'
    else:
        output_file = 'sandbox_matmul_results.json'
    with open(output_file, 'r') as f:
        data = json.load(f)
        matrix_sizes = data[0]
        speeds = data[1]
        f.close()
    return matrix_sizes, speeds

def triton_perf(M, N, K, AT=False, BT=False, dtype=torch.float16, warmup=25, rep=75):
    a = torch.rand((K, M) if AT else (M, K), device="cuda", dtype=dtype)
    b = torch.rand((N, K) if BT else (K, N), device="cuda", dtype=dtype)
    if AT:
        a = a.t()
    if BT:
        b = b.t()
    gflops = lambda ms: 2. * M * N * K / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton.ops.matmul(a, b), warmup=warmup, rep=rep)
    return gflops(ms)

def _worker_init(result_dir, env):
    global _result_dir, _env, _num_tasks, _done_tasks
    print('worker init')
    _result_dir = result_dir
    _env = env
    cmd = f'mkdir qdrep'
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, check=True, cwd=_result_dir)

def do_permutations(jobs, perms, bin_path, result_dir, env):
    num_tasks = len(perms)
    speeds = np.zeros((num_tasks,))
    runtimes = np.zeros((num_tasks,))
    async_results = [None] * num_tasks
    done_tasks = 0
    def callback(job_values):
        nonlocal done_tasks
        index, speed, runtime = job_values
        runtimes[index] = runtime
        done_tasks += 1
        if speed is False:
            print(f'{done_tasks}/{num_tasks} done, {perms[index]} took {runtime} and FAILED!')
        else:
            speeds[index] = speed
            print(f'{done_tasks}/{num_tasks} done, {perms[index]} took {runtime} and yields {speed}')

    with Pool(jobs, _worker_init, (result_dir, env)) as pool:
        for i, perm in enumerate(perms):
            if type(perm) == str:
                perm_name = perm.split('_')[1]
                matrix_size = perm.split('_')[2].split('.')[0]
            else:
                perm_name = perm.stem.split('_')[1]
                matrix_size = perm.stem.split('_')[2]
            #enable if you want nsys output
            #if perm_name in ['tvmcuda', 'ireecuda', 'mlircuda', 'cublas']:
            #    async_results[i] = pool.apply_async(_gpu_nsys_permutation, (i, bin_path / perm, matrix_size, perm_name), callback=callback)
            #else:
            async_results[i] = pool.apply_async(_do_single_permutation, (i, bin_path / perm, matrix_size), callback=callback)
        print("Submitted all jobs to pool")
        for ar in async_results:
            ar.get()
        pool.close()
        pool.join()

    return speeds

def main(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(argv[1:])

    result_dir = make_result_dir(args.results)

    write_system_info(result_dir, args.bins.parent / 'cpuinfo-install')

    # get only the executables
    bin_paths = [x for x in args.bins.iterdir() if
                x.is_file() and x.stat().st_mode & 0o111 and x.name.startswith('matmul')]

    # run iree-llvm-sandbox using python api
    if args.sandbox:
        build_path = args.bins.parent.absolute()
        os.environ["PYTHONPATH"] = os.path.join(build_path, "mlir/tools/iree_llvm_sandbox/python_package")
        os.environ["MLIR_RUNNER_UTILS_LIB"] = os.path.join(build_path, "mlir/lib/libmlir_runner_utils.so")
        os.environ["MLIR_C_RUNNER_UTILS_LIB"] = os.path.join(build_path, "mlir/lib/libmlir_c_runner_utils.so")
        try:
            cmd = 'cp iree_sandbox_matmul.py ./external/iree-llvm-sandbox/python/examples/matmul'
            subprocess.run(cmd, shell=True, check=True)
        except Exception:
            print("Error copying iree_sandbox_matmul.py")
            raise

        if args.benchmark_path:
            file_path = os.path.join(os.getcwd(), args.benchmark_path)
            sandbox_sizes, sandbox_speeds = sandbox_perf(file_path, args.num_iters)
        elif args.sandbox_configs:
            file_path = args.sandbox_configs
            sandbox_sizes, sandbox_speeds = sandbox_perf(file_path, args.num_iters, use_configs=True)

        if args.nodai_configs:
            file_path = args.nodai_configs
            nodai_sandbox_sizes, nodai_sandbox_speeds = sandbox_perf(file_path, args.num_iters, use_configs=True)

    # run triton using python api
    if args.triton:
        if args.benchmark_path:
            triton_sizes = []
            triton_speeds = []
            with open(args.benchmark_path, 'r') as f:
                all_sizes = f.readlines()
                f.close()

            for line in all_sizes:
                if line[0] == '#':
                    continue
                print("Triton running matmul size:", line)
                m_size = [int(x) for x in line.split('x')]

                if args.dtype == 'fp32':
                    triton_dtype = torch.float32
                elif args.dtype == 'fp16':
                    triton_dtype = torch.float16
                else:
                    raise ValueError(args.dtype, "is not supported.")

                speed = triton_perf(m_size[0], m_size[1], m_size[2], dtype=triton_dtype)
                triton_sizes.append(m_size)
                triton_speeds.append(speed)
        else:
            raise ValueError("Benchmark sizes are not provided!")

    # run them in parallel and collect the results
    speeds = do_permutations(args.jobs, list(x.name for x in bin_paths), args.bins, result_dir, BENCHMARK_ENV)
    # break up and interpret the file names
    binaries = {}
    for i, path in enumerate(bin_paths):
        parts = path.name.split('_')[1:]
        parts[1] = parts[1].split('.')[0]
        size = tuple(int(y) for y in parts[1].split('x'))
        binaries.setdefault(parts[0], []).append(
            {'path': path.resolve(), 'size': size, 'speed': speeds[i]})

    if args.sandbox:
        if args.benchmark_path or args.sandbox_configs:
            for i, size in enumerate(sandbox_sizes):
                binaries.setdefault('mlir-sandbox', []).append(
                    {'path': '', 'size': tuple(size), 'speed': sandbox_speeds[i]})
        if args.nodai_configs:
            for i, size in enumerate(nodai_sandbox_sizes):
                binaries.setdefault('nodai-mlir-sandbox', []).append(
                    {'path': '', 'size': tuple(size), 'speed': nodai_sandbox_speeds[i]})

    if args.triton:
        for i, size in enumerate(triton_sizes):
            binaries.setdefault('triton', []).append(
                {'path': '', 'size': tuple(size), 'speed': triton_speeds[i]})

    # used to impose a consistent sorting of the matrix sizes in the plot
    bar_ordering = list(collections.OrderedDict.fromkeys(y['size'] for x in binaries for y in binaries[x]))
    bar_ordering.sort(key=lambda s: (reduce(lambda x, y: x*y, s), s))

    any_error = False

    for idx, backend in enumerate(binaries):
        bar_x = []
        speeds = []
        for binary in binaries[backend]:
            print(backend, binary)
            speeds.append(binary['speed'])
            bar_x.append(bar_ordering.index(binary['size']) + idx * BAR_WIDTH)
        if len(bar_x) > 0:
            autolabel(plt.bar(bar_x, speeds, BAR_WIDTH, color=BAR_COLORS[backend], label=backend))
        else:
            print("No results could be collected for backend", backend)

    plt.xlabel("Matrix sizes")
    plt.ylabel("GFLOPS")
    plt.title("Single Precision Matrix Multiplication")

    system_info = ""
    f=open(result_dir / 'arch-info')
    lines=f.readlines()
    system_info = system_info + "CPU:{}: {} (cores x Microarch)".format(lines[1].strip(), lines[3].strip())
    f.close()

    gpu_info_file = Path(result_dir / 'gpu-info')
    if gpu_info_file.exists():
        f=open(gpu_info_file)
        lines=f.readlines()
        system_info = system_info + ", GPU Model:{}".format(lines[0].strip())
        f.close()
    system_info = system_info + ", " + args.dtype
    plt.suptitle(system_info, fontsize=7)

    x_pos = [i + 0.5*(len(binaries) - 1)*BAR_WIDTH for i in range(len(bar_ordering))]
    plt.xticks(x_pos, ['x'.join(str(d) for d in s) for s in bar_ordering], rotation=90, fontsize=5)
    plt.legend(loc='best')
    plt.savefig(result_dir / 'matmul.png', dpi=300, bbox_inches='tight')

    if any_error:
        print("Some benchmarks had problems, see above.")
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
