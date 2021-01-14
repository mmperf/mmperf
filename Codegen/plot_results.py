#!/usr/bin/python3

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os

sizes = ["18x32x96", "24x64x96" ,"24x64x512" ,"48x64x128" ,"192x64x128" , \
  "192x128x128" ,"192x256x256" ,"384x256x256" ,"480x512x16" ,"480x512x256" , \
  "1024x1024x1024" ,"1020x1152x1152" ,"1920x2304x2304" ,"2304x2304x2560" \
]
width = 0.35

def plot_mkl():
    gflops = []
    for size in sizes:
        file_name = 'mkl/matmul_' + size + '_mkl_perf.out'
        with open(file_name, 'r') as f:
            speed = f.readlines()
        gflops.append(float(speed[0].split()[0]))

    x_pos = [i for i, _ in enumerate(sizes)]
    plt.bar(x_pos, gflops, width, color='red', label='MKL')

def plot_mlir():
    gflops = []
    for size in sizes:
        file_name = 'build/matmul/matmul_' + size + '_mlir_perf.out'
        try:
            with open(file_name, 'r') as f:
                speed = f.readlines()
            gflops.append(float(speed[0].split()[0]))
        except:
            gflops.append(0.0)

    x_pos = [i + width for i, _ in enumerate(sizes)]
    plt.bar(x_pos, gflops, width, color='blue', label='MLIR')

if __name__ == "__main__":
    plot_mkl()
    plot_mlir()

    plt.xlabel("Matrix sizes")
    plt.ylabel("GFLOPS")
    plt.title("Single Precision Matrix Multiplication")
    x_pos = [i + 0.5 * width for i, _ in enumerate(sizes)]
    plt.xticks(x_pos, sizes, rotation=90, fontsize=5)
    plt.legend(loc='best')
    plt.savefig('matmul.png', dpi=300, bbox_inches='tight')
