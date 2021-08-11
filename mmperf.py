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

plt.style.use('ggplot')

BAR_WIDTH = 0.15
BAR_COLORS = {'mkl': 'cornflowerblue',
              'accelerate': 'lightgray',
              'mlir': 'sandybrown',
              'mlircuda': 'green',
              'openblas': 'mediumseagreen',
              'blis': 'mediumspringgreen',
              'blasfeo': 'olivedrab',
              'cublas': 'chocolate',
              'halide': 'gold',
              'ruy': 'violet',
              'tvm' : 'indigo',
              'naive': 'black',
              'nodai': 'red',
              'nodai-1': 'dodgerblue',
              'nodai-2': 'purple',
              'ireevmvx': 'thistle',
              'ireedylib': 'aqua',
              'ireecuda': 'deeppink'}
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
    parser.add_argument('bins', type=path_expand, help='Path where the test binaries are')
    parser.add_argument('results', type=path_expand, help='Result directory')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Number of parallel jobs for running the benchmarks')

def make_result_dir(base_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    result_dir = (base_dir / timestamp).resolve()
    os.makedirs(result_dir)
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

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        # If the floor value of GFLOPS is 0 print its float value
        if(int(height) is 0):
            plt.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%.3f' % float(height), fontsize=5, ha='center', va='bottom')
        else:
            plt.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    '%d' % int(height), fontsize=5, ha='center', va='bottom')

_result_dir = None
_env = None

def _do_single_permutation(i, path, duration=None):
    t0 = time.perf_counter()
    if not duration:
        duration = 1
    reps = str(duration)
    result = subprocess.run([str(path), f'{reps}'], stdout=subprocess.DEVNULL, cwd=_result_dir, env=_env, check=False)
    t1 = time.perf_counter()
    gflops_path = _result_dir / (path.name + '_perf.out')
    if gflops_path.is_file():
        speed = float(gflops_path.read_text().split()[0])
    runtime = t1 - t0
    if result.returncode != 0:
        print("Benchmark failed with error code:",
              signal.Signals(-result.returncode) if result.returncode < 0
              else result.returncode)
        return i, False, runtime
    else:
        return i, speed, runtime

def _worker_init(result_dir, env):
    global _result_dir, _env, _num_tasks, _done_tasks
    print('worker init')
    _result_dir = result_dir
    _env = env

def do_permutations(jobs, perms, bin_path, result_dir, env, duration=None):
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
            async_results[i] = pool.apply_async(_do_single_permutation, (i, bin_path / perm, duration), callback=callback)
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

    # run them in parallel and collect the results
    speeds = do_permutations(args.jobs, list(x.name for x in bin_paths), args.bins, result_dir, BENCHMARK_ENV)
    # break up and interpret the file names
    binaries = {}
    for i, path in enumerate(bin_paths):
        parts = path.name.split('_')[1:]
        parts[1] = parts[1].replace('m', '', 1)
        size = tuple(int(y) for y in parts[1].split('x'))
        binaries.setdefault(parts[0], []).append(
            {'path': path.resolve(), 'size': size, 'speed': speeds[i]})

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

    f=open(result_dir / 'arch-info')
    lines=f.readlines()
    plt.suptitle("CPU:%s: %s (cores x Microarch)" % (lines[1].strip(), lines[3].strip()), fontsize=8)

    x_pos = [i + 0.5*(len(binaries) - 1)*BAR_WIDTH for i in range(len(bar_ordering))]
    plt.xticks(x_pos, ['x'.join(str(d) for d in s) for s in bar_ordering], rotation=90, fontsize=5)
    plt.legend(loc='best')
    plt.savefig(result_dir / 'matmul.png', dpi=300, bbox_inches='tight')

    if any_error:
        print("Some benchmarks had problems, see above.")
    return result_dir

if __name__ == '__main__':
    sys.exit(main(sys.argv))
