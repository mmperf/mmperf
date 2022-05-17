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
import subprocess

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
      Tile('matmul',
           'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'],
           peel=[0, 1, 2])
        .then(Vectorize('matmul', ''))
        .then(LoweringOnlyExpert('matmul', 'linalg.generic'))
    ]]
  return all_experts, configs

def singleExpert3DPeel(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox SingleTiling3DPeel
    configs[0]['tile_sizes'] = [12, 32, 16]
    configs[0]['tile_interchange'] = [0, 1, 2]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      Tile('matmul',
           'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'],
           peel=[0, 1, 2])
        .then(Vectorize('matmul', ''))
        .then(LoweringOnlyExpert('matmul', 'linalg.generic'))
    ]]
  return all_experts, configs

def singleExpert3DPad(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox SingleTiling3DPad
    configs[0]['tile_sizes'] = [12, 32, 16]
    configs[0]['tile_interchange'] = [0, 1, 2]

  if default == True or 'hoist_padding' not in configs[0]:
    configs[0]['hoist_padding'] = [2, 3, 0]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      Tile('matmul',
           'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'])
        .then(Pad('matmul', 'linalg.generic',
                  pack_paddings=[1, 1, 0],
                  hoist_paddings=configs[0]['hoist_padding']))
        .then(Vectorize('matmul', ''))
        .then(LoweringOnlyExpert('matmul', 'linalg.generic'))
    ]]
  return all_experts, configs

def singleExpert3DPeelTranspose(configs, default=False):
  if default == True:
    # Use default config values from iree-llvm-sandbox SingleTiling3DPad
    configs[0]['tile_sizes'] = [6, 32, 16]
    configs[0]['tile_interchange'] = [2, 1, 0]

  all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
      Tile('matmul',
           'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'],
           peel=[0, 1, 2])
        .then(Vectorize('matmul', ''))
        .then(LoweringOnlyExpert('matmul',
                                 'linalg.generic',
                                 transpose_lowering='shuffle'))
    ]]
  return all_experts, configs

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
      Tile('matmul', 'linalg.generic',
           tile_sizes=configs[0]['tile_sizes'],
           tile_interchange=configs[0]['tile_interchange'])
          .then(Tile('matmul', 'linalg.generic',
                     tile_sizes=configs[1]['tile_sizes'],
                     tile_interchange=configs[1]['tile_interchange']))
          .then(Pad('matmul', 'linalg.generic',
                    pack_paddings=[1, 1, 0],
                    hoist_paddings=configs[0]['hoist_padding'],
                    transpose_paddings=[[1, 0], [0, 1], [0, 1]]))
          .then(Vectorize('matmul', ''))
          .then(UnrollOneParentLoop('matmul',
                                  'vector.contract',
                                  parent_loop_num=1,
                                  unroll_factor=4))
          .then(LoweringOnlyExpert('matmul',
                                 'linalg.generic',
                                 transpose_lowering='eltwise'))
    ]]
  return all_experts, configs

def generate_config_options(configs, matrix_size, expert_name, matmul_spec, dynamic_compile):
  if expert_name == "DoubleTile2DPadAndHoist":
    options = [{
        "tile_sizes": configs[0]['tile_sizes'],
        "tile_interchange": configs[0]['tile_interchange']
      },
      {
        "tile_sizes": configs[1]['tile_sizes'],
        "tile_interchange": configs[1]['tile_interchange']
      }
    ]
  else:
    options = [{
      "tile_sizes": configs[0]['tile_sizes'],
      "tile_interchange": configs[0]['tile_interchange']
    }]

  return {
    "options": options,
    "identifier": "matmul",
    "expert": expert_name,
    "compile": dynamic_compile,
    "spec": matmul_spec,
    "m": matrix_size[0],
    "n": matrix_size[1],
    "k": matrix_size[2]
  }

def path_expand(s):
  return Path(s).expanduser().resolve()

def matrix_size_string(input):
  if isinstance(input, dict):
    return f'{input["m"]}x{input["n"]}x{input["k"]}'
  elif isinstance(input, tuple):
    return 'x'.join([str(d) for d in input])
  else:
    return None

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']

# CHECK-NOT: FAILURE
def main(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-matrix_path', type=path_expand, help='Path to file containing matrix sizes to be run')
  parser.add_argument('-config_path', type=path_expand, help='Path to load config file')
  parser.add_argument('-n_iters', type=int, default=100, help='Number of iterations to run matmul')
  parser.add_argument('-save_dir', default='../../mlir_sandbox_configs', help='Path to save config files')
  parser.add_argument('-obj_dir', default='../../mlir_sandbox_objs', help='Path to save config files')
  args = parser.parse_args(argv[1:])

  cmd = f'mkdir -p {args.save_dir}'
  subprocess.run(cmd, shell=True, check=True)
  cmd = f'mkdir -p {args.obj_dir}'
  subprocess.run(cmd, shell=True, check=True)

  expert_list = ["SingleTiling2DPeel",
                  "SingleTiling3DPeel",
                  "SingleTiling3DPad",
                  "SingleTiling3DPeelTranspose",
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

    expert1, configs1 = singleExpert2DPeel([{}, {}], True)
    expert2, configs2 = singleExpert3DPeel([{}, {}], True)
    expert3, configs3 = singleExpert3DPad([{}, {}], True)
    expert4, configs4 = singleExpert3DPeelTranspose([{}, {}], True)
    expert5, configs5 = doubleExpert2DPad([{}, {}], True)

    all_experts = expert1 + expert2 + expert3 + expert4 + expert5
    all_configs = [configs1, configs2, configs3, configs4, configs5]

    for line in all_sizes:
      if line[0] == '#':
        continue
      m_size = [int(x) for x in line.split('x')]
      matrix_sizes.append(m_size)

      experts_2, speeds_2 = [], []
      for dynamic_at_compile_time in dynamic_at_compile_time_list:
        experts_1, speeds_1 = [], []
        for spec in spec_list:
          spec_name = spec.replace(',', '')
          compile_name = compile_time_name_list[dynamic_at_compile_time_list.
                            index(dynamic_at_compile_time)].replace(' ', '')
          obj_file_name = matrix_size_string(tuple(m_size)) + "_" + spec_name + "_" + compile_name + ".o"
          obj_file_path = os.path.join(args.obj_dir, obj_file_name)
          results = test_harness(lambda s, t: EinsumProblem(spec, 'mnk', 2),
                                 [[np.float32] * 3],
                                 test_sizes(keys, [m_size]),
                                 test_experts(all_experts, expert_list),
                                 dynamic_at_compile_time_sizes=set(
                                     dynamic_at_compile_time).intersection(keys),
                                 n_iters=args.n_iters,
                                 function_name='matmul',
                                 dump_obj_to_file=obj_file_path)

          expert_gflops = results.data['gflop_per_s_per_iter'][int(args.n_iters / 2)].values.tolist()
          max_gflops = max(expert_gflops)
          max_expert = expert_list[expert_gflops.index(max_gflops)]
          speeds_1.append(max_gflops)
          experts_1.append(max_expert)
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

      best_expert = experts_2[max_speeds_idx]
      config = all_configs[expert_list.index(best_expert[0])]
      output_file = generate_config_options(config, m_size, best_expert[0], best_expert[1], best_expert[2])
      best_expert[1] = best_expert[1].replace(',', '')
      best_expert[2] = best_expert[2].replace(' ', '')

      output_path = Path(args.save_dir) / f'{matrix_size_string(tuple(m_size))}_' \
                                          f'{best_expert[0]}_{best_expert[1]}_{best_expert[2]}.json'
      with output_path.open('w') as of:
        of.write(json.dumps(output_file))
      print("Config file is saved to", output_path)

    with open('../../sandbox_matmul_results.json', 'w') as f:
      json.dump([matrix_sizes, speeds, experts], f)
      f.close()

  elif args.config_path:
    all_results = dict()
    for f_path in glob.glob(os.path.join(args.config_path, '*.json')):
      with open(f_path, 'r') as f:
        data = json.load(f)
        matrix_size = [int(data["m"]), int(data["n"]), int(data["k"])]
        m_size_str = f'{matrix_size[0]}x{matrix_size[1]}x{matrix_size[2]}'
        configs = data["options"]
        expert_name = data["expert"]
        spec = data["spec"]
        compile_name = data["compile"]
        compile_type = dynamic_at_compile_time_list[compile_time_name_list.index(compile_name)]
        f.close()

      if expert_name == "SingleTiling2DPeel":
        all_experts, _ = singleExpert2DPeel(configs)
      elif expert_name == "SingleTiling3DPeel":
        all_experts, _ = singleExpert3DPeel(configs)
      elif expert_name == "SingleTiling3DPad":
        all_experts, _ = singleExpert3DPad(configs)
      elif expert_name == "SingleTiling3DPeelTranspose":
        all_experts, _ = singleExpert3DPeelTranspose(configs)
      elif expert_name == "DoubleTile2DPadAndHoist":
        all_experts, _ = doubleExpert2DPad(configs)

      spec_name = spec.replace(',', '')
      dynamic_compile_name = compile_name.replace(' ', '')
      obj_file_name = m_size_str + "_" + expert_name + "_" + spec_name + "_" + dynamic_compile_name + ".o"
      obj_file_path = os.path.join(args.obj_dir, obj_file_name)
      results = test_harness(lambda s, t: EinsumProblem(spec, 'mnk', 2),
                             [[np.float32] * 3],
                             test_sizes(keys, [matrix_size]),
                             test_experts(all_experts, [expert_name]),
                             dynamic_at_compile_time_sizes=compile_type,
                             n_iters=args.n_iters,
                             function_name='matmul',
                             dump_obj_to_file=obj_file_path)

      expert_gflops = results.data['gflop_per_s_per_iter'][int(args.n_iters/2)]
      key_str = str(matrix_size)
      if key_str in all_results:
        all_results[key_str] += [[expert_gflops, expert_name, spec, compile_name]]
      else:
        all_results[key_str] = [[expert_gflops, expert_name, spec, compile_name]]

    for key, value in all_results.items():
      matrix_sizes.append(json.loads(key))
      max_expert = max(value)
      speeds.append(max_expert[0])
      experts.append(max_expert[1:])

    with open('../../nodai_sandbox_matmul_results.json', 'w') as f:
      json.dump([matrix_sizes, speeds, experts], f)
      f.close()

  else:
    raise ValueError("Please input matrix_path or config path")

if __name__ == '__main__':
  sys.exit(main(sys.argv))
