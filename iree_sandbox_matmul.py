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

def singleExpert2DPeel(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox SingleTiling2DPeel
    configs[0]['tile_sizes'] = [6, 32, 1]
    configs[0]['tile_interchange'] = [0, 1, 2]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      Tile('matmul_on_tensors',
           'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'],
           peel=[0, 1, 2])
        .then(Vectorize('matmul_on_tensors', ''))
        .then(LoweringOnlyExpert('matmul_on_tensors', 'linalg.generic'))
    ]]
  return all_experts

def singleExpert3DPeel(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox SingleTiling3DPeel
    configs[0]['tile_sizes'] = [12, 32, 16]
    configs[0]['tile_interchange'] = [0, 1, 2]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      Tile('matmul_on_tensors',
           'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'],
           peel=[0, 1, 2])
        .then(Vectorize('matmul_on_tensors', ''))
        .then(LoweringOnlyExpert('matmul_on_tensors', 'linalg.generic'))
    ]]
  return all_experts

def singleExpert3DPad(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox SingleTiling3DPad
    configs[0]['tile_sizes'] = [12, 32, 16]
    configs[0]['tile_interchange'] = [0, 1, 2]

  if default == True or 'hoist_padding' not in configs[0]:
    configs[0]['hoist_padding'] = [2, 3, 0]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      Tile('matmul_on_tensors',
           'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'],
           pad=True,
           pack_paddings=[1, 1, 0],
           hoist_paddings=configs[0]['hoist_padding'])
        .then(Vectorize('matmul_on_tensors', ''))
        .then(LoweringOnlyExpert('matmul_on_tensors', 'linalg.generic'))
    ]]
  return all_experts

def doubleExpert2DPad(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox DoubleTile2DPadAndHoist
    configs[0]['tile_sizes'] = [288, 128, 512]
    configs[0]['tile_interchange'] = [0, 2, 1]
    configs[1]['tile_sizes'] = [12, 32, 1]
    configs[1]['tile_interchange'] = [0, 1, 2]

  if default == True or 'hoist_padding' not in configs[0]:
    configs[0]['hoist_padding'] = [5, 6, 0]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      DoubleTile('matmul_on_tensors',
                 'linalg.generic',
                 tile_sizes1=configs[0]['tile_sizes'],
                 tile_interchange1=configs[0]['tile_interchange'],
                 tile_sizes2=configs[1]['tile_sizes'],
                 tile_interchange2=configs[1]['tile_interchange'],
                 pad2=True,
                 pack_paddings2=[1, 1, 0],
                 hoist_paddings2=configs[0]['hoist_padding'],
                 transpose_paddings2=[[1, 0], [0, 1], [0, 1]],)
        .then(Vectorize('matmul_on_tensors', ''))
        .then(UnrollOneParentLoop('matmul_on_tensors',
                                  'vector.contract',
                                  parent_loop_num=1,
                                  unroll_factor=4))
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
  parser.add_argument('-n_iters', type=int, default=100, help='Number of iterations to run matmul')
  args = parser.parse_args(argv[1:])

  expert_list = ["SingleTiling2DPeel",
                  "SingleTiling3DPeel",
                  "SingleTiling3DPad",
                  "DoubleTile2DPadAndHoist"]
  dynamic_at_compile_time_list = [[],  # case 1: static at compile time
                                  ['m', 'k'],  # case 2: partially dynamic at compile time
                                  keys]  # case 3: fully dynamic at compile time
  compile_time_name_list = ['static', 'partially dynamic', 'fully dynamic']
  spec_list = [
    'km,kn',  # C += A^T.B  fastest
    'mk,kn',  # C += A.B
    'mk,nk'  # C += A.B^T  slowest
  ]
  
  speeds = []
  experts = []
  matrix_sizes = []

  if args.matrix_path:
    with open(args.matrix_path, 'r') as f:
      all_sizes = f.readlines()
      f.close()

    all_experts = singleExpert2DPeel([{}, {}], True) + \
                  singleExpert3DPeel([{}, {}], True) + \
                  singleExpert3DPad([{}, {}], True) + \
                  doubleExpert2DPad([{}, {}], True)

    for line in all_sizes:
      if line[0] == '#':
        continue
      m_size = [int(x) for x in line.split('x')]
      matrix_sizes.append(m_size)

      experts_2, speeds_2 = [], []
      for dynamic_at_compile_time in dynamic_at_compile_time_list:
        experts_1, speeds_1 = [], []
        for spec in spec_list:
          results = test_harness(lambda s, t: EinsumProblem(spec, 'mnk', 2),
                                 [[np.float32] * 3],
                                 test_sizes(keys, [m_size]),
                                 test_experts(all_experts, expert_list),
                                 dynamic_at_compile_time_sizes=set(
                                     dynamic_at_compile_time).intersection(keys),
                                 n_iters=args.n_iters,
                                 function_name='matmul_on_tensors')

          expert_gflops = results.data['gflop_per_s_per_iter'][int(args.n_iters/2)].values.tolist()
          max_gflops = max(expert_gflops)
          speeds_1.append(max_gflops)
          experts_1.append(expert_list[expert_gflops.index(max_gflops)])
        max_speeds_1 = max(speeds_1)
        max_speeds_idx = speeds_1.index(max_speeds_1)
        speeds_2.append(max_speeds_1)
        experts_2.append([experts_1[max_speeds_idx], spec_list[max_speeds_idx]])
      max_speeds_2 = max(speeds_2)
      max_speeds_idx = speeds_2.index(max_speeds_2)
      speeds.append(max_speeds_2)
      experts_2[max_speeds_idx].append(compile_time_name_list[max_speeds_idx])
      experts.append(experts_2[max_speeds_idx])
      print("Best speed: ", max_speeds_2, experts_2[max_speeds_idx])

    with open('../../sandbox_matmul_results.json', 'w') as f:
      json.dump([matrix_sizes, speeds, experts], f)
      f.close()

  elif args.config_path:
    for f_path in glob.glob(os.path.join(args.config_path, '*.json')):
      with open(f_path, 'r') as f:
        data = json.load(f)
        matrix_size = [int(data["m"]), int(data["n"]), int(data["k"])]
        configs = data["options"]
        expert_name = data["expert"]
        spec = data["spec"]
        compile_name = data["compile"]
        compile_type = dynamic_at_compile_time_list[compile_time_name_list.index(compile_name)]
        matrix_sizes.append(matrix_size)
        f.close()

      if expert_name == "SingleTiling2DPeel":
        all_experts = singleExpert2DPeel(configs)
      elif expert_name == "SingleTiling3DPeel":
        all_experts = singleExpert3DPeel(configs)
      elif expert_name == "SingleTiling3DPad":
        all_experts = singleExpert3DPad(configs)
      elif expert_name == "DoubleTile2DPadAndHoist":
        all_experts = doubleExpert2DPad(configs)

      results = test_harness(lambda s, t: EinsumProblem(spec, 'mnk', 2),
                             [[np.float32] * 3],
                             test_sizes(keys, [matrix_size]),
                             test_experts(all_experts, [expert_name]),
                             dynamic_at_compile_time_sizes=compile_type,
                             n_iters=args.n_iters,
                             function_name='matmul_on_tensors')

      expert_gflops = results.data['gflop_per_s_per_iter'][int(args.n_iters/2)]
      speeds.append(expert_gflops)
      experts.append([expert_name, spec, compile_name])

    with open('../../nodai_sandbox_matmul_results.json', 'w') as f:
      json.dump([matrix_sizes, speeds, experts], f)
      f.close()

  else:
    raise ValueError("Please input matrix_path or config path")

if __name__ == '__main__':
  sys.exit(main(sys.argv))
