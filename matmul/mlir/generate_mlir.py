#!/usr/bin/python3
import argparse

code = '''\
func @matmul(%a: memref<{M}x{K}xf32>, %b: memref<{K}x{N}xf32>, %c: memref<{M}x{N}xf32>) {{
  linalg.matmul ins(%a, %b : memref<{M}x{K}xf32>, memref<{K}x{N}xf32>)
    outs(%c: memref<{M}x{N}xf32>)
  return
}}'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-matrix_sizes', dest='matrix_sizes', action='store', help='Path to file containing matrix sizes',
                        default='../../benchmark_sizes.txt')
    args = parser.parse_args()

    sizes = None
    with open(args.matrix_sizes, 'r') as f:
        sizes = f.read().splitlines()
    for size in sizes:
        dims = size.split('x')
        M = int(dims[0])
        N = int(dims[1])
        K = int(dims[2])
        args = {'M':M, 'N':N, 'K':K}
        with open('matmul_' + size + '.mlir', 'w') as f:
            f.write(code.format(**args))
