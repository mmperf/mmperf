#!/usr/bin/python3

sizes = ["18x32x96", "24x64x96" ,"24x64x512" ,"48x64x128" ,"192x64x128" , \
  "192x128x128" ,"192x256x256" ,"384x256x256" ,"480x512x16" ,"480x512x256" , \
  "1024x1024x1024" ,"1020x1152x1152" ,"1920x2304x2304" ,"2304x2304x2560" \
]

code = '''\
func @matmul(%a: memref<{M}x{K}xf32>, %b: memref<{K}x{N}xf32>, %c: memref<{M}x{N}xf32>) {{
  linalg.matmul ins(%a, %b : memref<{M}x{K}xf32>, memref<{K}x{N}xf32>)
    outs(%c: memref<{M}x{N}xf32>)
  return
}}'''

for size in sizes:
    dims = size.split('x')
    M = int(dims[0])
    N = int(dims[1])
    K = int(dims[2])
    args = {'M':M, 'N':N, 'K':K}
    with open('matmul_' + size + '.mlir', 'w') as f:
        f.write(code.format(**args))
