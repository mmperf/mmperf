import argparse
from tune_matmul import *
import sys


def main(argv):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('-m', '-matrix_sizes', dest='matrix_sizes', action='store', required=True,
                        help='Path to file containing matrix sizes to be run')
    parser.add_argument('-target', dest='target', action='store', help='Backend to run on',
                        default='llvm -mcpu=skylake-avx512')
    parser.add_argument('-num_threads', type=int, default=1, help='The number of physical CPU cores')
    args = parser.parse_args()
    os.environ["TVM_NUM_THREADS"] = str(args.num_threads)

    with open(args.matrix_sizes, 'r') as f:
        all_sizes = f.readlines()

    for line in all_sizes:
        if line[0] != '#':
            dims = [int(x) for x in line.split('x')]
            autotune(dims[0], dims[1], dims[2], args.target)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
