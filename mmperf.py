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
from pathlib import Path
from functools import reduce

def add_arguments(parser):
    parser.add_argument('bins', type=Path, help='Path where the test binaries are')
    parser.add_argument('results', type=Path, help='Result directory')

def main(argv):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args(argv[1:])

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    #result_dir = os.path.abspath(os.path.join(args.results, timestamp))
    result_dir = (args.results / timestamp).resolve()
    os.makedirs(result_dir)

    plt = platform.system()

    if plt == "Linux":
        # linux
        print("Linux System Detected.. looking for /proc/cpuinfo")
        shutil.copyfile(Path("/proc/cpuinfo"), result_dir / "cpuinfo")

        cpu_pattern = re.compile('cpu[0-9]+')
        cpudirs = [x for x in Path("/sys/devices/system/cpu/").iterdir() if cpu_pattern.match(x.name)]
        with open(result_dir / 'scaling_governor', 'w') as f:
            for cpu in cpudirs:
                f.write(cpu.name + ": " + (cpu / 'cpufreq' / 'scaling_governor').read_text())
        cpudirs = [x for x in Path("/sys/devices/system/cpu/").iterdir() if cpu_pattern.match(x.name)]
        with open(result_dir / 'core_frequencies', 'w') as f:
            for cpu in cpudirs:
                f.write(cpu.name + ": " + (cpu / 'cpufreq' / 'scaling_cur_freq').read_text())
    elif plt == "Darwin":
        # OSX
        print("OSX System Detected")
    else:
        print("Unidentified system")

    # get only the executables
    binaries = [x for x in Path(args.bins).iterdir() if
                x.is_file() and x.stat().st_mode & 0o111 and x.name.startswith('matmul')]
    # break up and intpret the file names
    binaries = [(x.resolve(), x.name.split('_')[1:]) for x in binaries]
    binaries = [(x[0], x[1][0], tuple(int(y) for y in x[1][1].split('x'))) for x in binaries]
    # sort by backend, then the product of the dimensions, then the dimensions
    binaries.sort(key=lambda x: (x[1], reduce(lambda x,y: x*y, x[2]), x[2]))

    my_env = os.environ.copy()
    my_env["MKL_NUM_THREADS"] = "1"
    my_env["OPENBLAS_NUM_THREADS"] = "1"
    my_env["HL_NUM_THREADS"] = "1"
    my_env["THREADS"] = "1"

    for b in binaries:
        print(b[0], b[1], b[2])
        subprocess.run([b[0]],cwd=result_dir, env=my_env)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
