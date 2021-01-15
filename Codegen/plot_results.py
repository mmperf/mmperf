#!/usr/bin/python3

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import argparse

width = 0.35

def plot_mkl(sizes, mkl_dir):
    gflops = []
    for size in sizes:
        file_name = mkl_dir + '/matmul_' + size + '_mkl_perf.out'
        with open(file_name, 'r') as f:
            speed = f.readlines()
        gflops.append(float(speed[0].split()[0]))

    x_pos = [i for i, _ in enumerate(sizes)]
    plt.bar(x_pos, gflops, width, color='red', label='MKL')

def plot_mlir(sizes, mlir_dir):
    gflops = []
    for size in sizes:
        file_name = mlir_dir + '/matmul_' + size + '_mlir_perf.out'
        try:
            with open(file_name, 'r') as f:
                speed = f.readlines()
            gflops.append(float(speed[0].split()[0]))
        except:
            gflops.append(0.0)

    x_pos = [i + width for i, _ in enumerate(sizes)]
    plt.bar(x_pos, gflops, width, color='blue', label='MLIR')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-matrix_sizes', dest='matrix_sizes', action='store',
                        help='Path to file containing matrix sizes', default='benchmark_sizes.txt')
    parser.add_argument('-mkl_dir', dest='mkl_dir', action='store',
                        help='Path to MKL performance results', default='mkl/')
    parser.add_argument('-mlir_dir', dest='mlir_dir', action='store',
                        help='Path to MLIR performance results', default='build/matmul/')
    args = parser.parse_args()

    sizes = None
    with open(args.matrix_sizes, 'r') as f:
        sizes = f.read().splitlines()

    plot_mkl(sizes, args.mkl_dir)
    plot_mlir(sizes, args.mlir_dir)

    plt.xlabel("Matrix sizes")
    plt.ylabel("GFLOPS")
    plt.title("Single Precision Matrix Multiplication")
    x_pos = [i + 0.5 * width for i, _ in enumerate(sizes)]
    plt.xticks(x_pos, sizes, rotation=90, fontsize=5)
    plt.legend(loc='best')
    plt.savefig('matmul.png', dpi=300, bbox_inches='tight')
