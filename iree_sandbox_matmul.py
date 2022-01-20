from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

import typing as tp
import json
import argparse
import sys
import os
import glob
from pathlib import Path

################################################################################
### Compilation strategies.
################################################################################

def singleExpert(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox SingleTiling3D
    configs[0]['tile_sizes'] = [12, 32, 16]
    configs[0]['tile_interchange'] = [0, 1, 2]

  if default == True or 'pack_padding' not in configs[0]:
    configs[0]['pack_padding'] = [1, 1, 0]
  if default == True or 'hoist_padding' not in configs[0]:
    configs[0]['hoist_padding'] = [2, 3, 0]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      SingleTilingExpert('matmul_on_tensors',
                         'linalg.generic',
                         tile_sizes=configs[0]['tile_sizes'],
                         tile_interchange=configs[0]['tile_interchange'],
                         pad=True,
                         pack_paddings=configs[0]['pack_padding'],
                         hoist_paddings=configs[0]['hoist_padding'])]]
  return all_experts

def doubleExpert(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox DoubleTileAndDecompose2DLarge
    configs[0]['tile_sizes'] = [128, 384, 512]
    configs[0]['tile_interchange'] = [0, 1, 2]
    configs[1]['tile_sizes'] = [12, 32, 1]
    configs[1]['tile_interchange'] = [1, 0, 2]
  
  if default == True or 'pack_padding' not in configs[0]:
    configs[0]['pack_padding'] = [1, 1, 0]
  if default == True or 'hoist_padding' not in configs[0]:
    configs[0]['hoist_padding'] = [3, 2, 0]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      DoubleTileAndDecompose('matmul_on_tensors',
                             'linalg.generic',
                             tile_sizes1=configs[0]['tile_sizes'],
                             tile_interchange1=configs[0]['tile_interchange'],
                             tile_sizes2=configs[1]['tile_sizes'],
                             tile_interchange2=configs[1]['tile_interchange'],
                             pad2=True,
                             pack_paddings2=configs[0]['pack_padding'],
                             hoist_paddings2=configs[0]['hoist_padding'])
        .then(Vectorize('matmul_on_tensors', 'linalg.generic'))
        .then(LoweringOnlyExpert('matmul_on_tensors',
                                 'linalg.generic',
                                 transpose_lowering='eltwise'))
    ]]
  return all_experts

def singleExpert2D():
  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      SingleTilingExpert('matmul_on_tensors',
                         'linalg.generic',
                         tile_sizes=[12, 32, 1],
                         tile_interchange=[0, 1, 2],
                         pad=True,
                         pack_paddings=[1, 1, 0],
                         hoist_paddings=[2, 3, 0])]]
  return all_experts

def doubleExpert2D():
  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      DoubleTileAndDecompose('matmul_on_tensors',
                             'linalg.generic',
                             tile_sizes1=[288, 128, 512],
                             tile_interchange1=[0, 2, 1],
                             tile_sizes2=[12, 32, 1],
                             tile_interchange2=[0, 1, 2],
                             pad2=True,
                             pack_paddings2=[1, 1, 0],
                             hoist_paddings2=[5, 6, 0])
        .then(Vectorize('matmul_on_tensors', 'linalg.generic'))
        .then(UnrollOneParentLoop('matmul_on_tensors',
                                  'vector.contract',
                                  parent_loop_num=1,
                                  unroll_factor=4))
        .then(LoweringOnlyExpert('matmul_on_tensors',
                                 'linalg.generic',
                                 transpose_lowering='eltwise'))
    ]]
  return all_experts

def doubleExpert3D():
  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      DoubleTileAndDecompose('matmul_on_tensors',
                             'linalg.generic',
                             tile_sizes1=[288, 128, 512],
                             tile_interchange1=[0, 2, 1],
                             tile_sizes2=[12, 32, 16],
                             tile_interchange2=[0, 1, 2],
                             pad2=True,
                             pack_paddings2=[1, 1, 0],
                             hoist_paddings2=[5, 6, 0])
        .then(Vectorize('matmul_on_tensors', 'linalg.generic'))
        .then(LoweringOnlyExpert('matmul_on_tensors',
                                 'linalg.generic',
                                 transpose_lowering='eltwise'))
    ]]
  return all_experts


################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']

def path_expand(s):
  return Path(s).expanduser().resolve()


# CHECK-NOT: FAILURE
def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-matrix_path', type=path_expand, help='Path to file containing matrix sizes to be run')
  parser.add_argument('-config_path', type=path_expand, help='Path to config file')
  args = parser.parse_args(argv[1:])

  n_iters = 100
  expert_lists = ["SingleTiling2D",
                  "SingleTiling3D",
                  "DoubleTileAndDecompose2D",
                  "DoubleTileAndDecompose3D",
                  "DoubleTileAndDecompose2DLarge"]
  speeds = []
  experts = []
  matrix_sizes = []

  if args.matrix_path:
    with open(args.matrix_path, 'r') as f:
      all_sizes = f.readlines()
      f.close()

    all_experts = singleExpert2D() + \
                  singleExpert([{}, {}], True) + \
                  doubleExpert2D() + \
                  doubleExpert3D() + \
                  doubleExpert([{}, {}], True)

    for line in all_sizes:
      if line[0] == '#':
        continue
      m_size = [int(x) for x in line.split('x')]
      matrix_sizes.append(m_size)

      results = test_harness(lambda s, t: EinsumProblem('km,kn', 2), [[np.float32] * 3],
                             test_sizes(keys, [m_size]),
                             all_experts,
                             n_iters=n_iters,
                             function_name='matmul_on_tensors')

      expert_gflops = results.data['gflop_per_s_per_iter'][50].values.tolist()
      max_gflops = max(expert_gflops)
      speeds.append(max_gflops)
      experts.append(expert_lists[expert_gflops.index(max_gflops)])

  elif args.config_path:
    for f_path in glob.glob(os.path.join(args.config_path, '*.json')):
      with open(f_path, 'r') as f:
        data = json.load(f)
        matrix_size = [int(data["m"]), int(data["n"]), int(data["k"])]
        configs = data["options"]
        matrix_sizes.append(matrix_size)
        f.close()

      if len(configs) == 1:
        all_experts = singleExpert(configs)
      else:
        all_experts = doubleExpert(configs)

      results = test_harness(lambda s, t: EinsumProblem('km,kn', 2), [[np.float32] * 3],
                      test_sizes(make_size_list, [matrix_size]),
                      all_experts,
                      n_iters=n_iters,
                      function_name='matmul_on_tensors')

      expert_gflops = results.data['gflop_per_s_per_iter'][50].values.tolist()
      max_gflops = max(expert_gflops)
      speeds.append(max_gflops)
      experts.append(expert_lists[expert_gflops.index(max_gflops)])
  else:
    raise ValueError("Please input matrix_path or config path")

  with open('../../sandbox_matmul_results.json', 'w') as f:
    json.dump([matrix_sizes, speeds, experts], f)
    f.close()

if __name__ == '__main__':
  sys.exit(main(sys.argv))
