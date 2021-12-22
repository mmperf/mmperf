from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

import typing as tp
import json
import argparse
import sys
from pathlib import Path

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
    SingleTilingExpert('matmul_on_tensors',
                       'linalg.generic',
                       tile_sizes=[12, 32, 8],
                       tile_interchange=[0, 1, 2],
                       pad=True,
                       pack_paddings=[1, 1, 0],
                       hoist_paddings=[2, 3, 0]),
    DoubleTileAndDecompose('matmul_on_tensors',
                            'linalg.generic',
                            tile_sizes1=[288, 128, 512],
                            tile_interchange1=[0, 2, 1],
                            tile_sizes2=[9, 32, 16],
                            tile_interchange2=[0, 1, 2],
                            pad2=True,
                            pack_paddings2=[1, 1, 0],
                            hoist_paddings2=[5, 6, 0]).then(\
      Vectorize('matmul_on_tensors',
                'linalg.generic')).then(\
      LoweringOnlyExpert('matmul_on_tensors',
                         'linalg.generic',
                         transpose_lowering='eltwise'))
    ]]

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']

def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}

def path_expand(s):
  return Path(s).expanduser().resolve()


# CHECK-NOT: FAILURE
def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('matrix_path', type=path_expand, help='Path to file containing matrix sizes to be run')
  args = parser.parse_args(argv[1:])

  n_iters = 100
  expert_lists = ["single_tiling_expert", "double_tile_and_decompose"]

  with open(args.matrix_path, 'r') as f:
    all_sizes = f.readlines()
    f.close()

  speeds = []
  experts = []
  matrix_sizes = []
  for line in all_sizes:
    if line[0] == '#':
      continue

    m_size = [int(x) for x in line.split('x')]
    matrix_sizes.append(m_size)

    results = test_harness(lambda s, t: EinsumProblem('mk,kn'), [[np.float32] * 3],
                    map(make_size_list, [m_size]),
                    all_experts,
                    n_iters=n_iters,
                    function_name='matmul_on_tensors')

    expert_gflops = []
    for key, value in results.items():
      expert_gflops.append(value['gflop_per_s_per_iter'][4])
    max_gflops = max(expert_gflops)
    speeds.append(max_gflops)
    experts.append(expert_lists[expert_gflops.index(max_gflops)])

  with open('../../sandbox_matmul_results.json', 'w') as f:
    json.dump([matrix_sizes, speeds, experts], f)
    f.close()

if __name__ == '__main__':
  sys.exit(main(sys.argv))