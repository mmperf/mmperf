import json
import argparse
from pathlib import Path
import subprocess

def sandbox_config_options(matrix_size, expert_name, matmul_spec, dynamic_compile):
  if expert_name == "SingleTiling2DPeel":
    options = [{
      "tile_sizes": [6, 16, 1],
      "tile_interchange": [0, 1, 2]
    }]
  elif expert_name == "SingleTiling3DPeel":
    options = [{
      "tile_sizes": [6, 16, 8],
      "tile_interchange": [0, 1, 2]
    }]
  elif expert_name == "SingleTiling3DPad":
    options = [{
      "tile_sizes": [6, 16, 8],
      "tile_interchange": [0, 1, 2]
    }]
  elif expert_name == "DoubleTile2DPadAndHoist":
    options = [{
        "tile_sizes": [288, 128, 512],
        "tile_interchange": [0, 2, 1]
      },
      {
        "tile_sizes": [6, 16, 1],
        "tile_interchange": [0, 1, 2]
      }
    ]

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

def matrix_size_string(input):
  if isinstance(input, dict):
    return f'{input["m"]}x{input["n"]}x{input["k"]}'
  elif isinstance(input, tuple):
    return 'x'.join([str(d) for d in input])
  else:
    return None

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-load_file', help='Path to results file')
  parser.add_argument('-save_dir', default='./sandbox_configs', help='Path to results file')
  parser.add_argument('-matrix_path', help='Path to file containing matrix sizes to be run')
  args = parser.parse_args()

  cmd = f'mkdir -p {args.save_dir}'
  subprocess.run(cmd, shell=True, check=True)

  if args.load_file:
    # Generate specific config for each matmul size from previously generated results
    with open(Path(args.load_file), 'r') as f:
        data = json.load(f)
        matrix_sizes = data[0]
        expert_info = data[2]
        f.close()
    for size, expert in zip(matrix_sizes, expert_info):
      config = sandbox_config_options(size, expert[0], expert[1], expert[2])
      expert[1] = expert[1].replace(',', '')
      expert[2] = expert[2].replace(' ', '')

      output_path = Path(args.save_dir) / f'{matrix_size_string(tuple(size))}_{expert[0]}_{expert[1]}_{expert[2]}.json'
      with output_path.open('w') as of:
        of.write(json.dumps(config))
      print("Config file is saved to", output_path)
  else:
    # Generate all possible combination of configs and use for naive search
    assert args.matrix_path != None, "Path to the input matmul sizes is needed!"
    with open(args.matrix_path, 'r') as f:
      all_sizes = f.readlines()
      f.close()

    expert_list = ["SingleTiling2DPeel",
                   "SingleTiling3DPeel",
                   "SingleTiling3DPad",
                   "DoubleTile2DPadAndHoist"]
    compile_time_name_list = ['static', 'partially dynamic', 'fully dynamic']
    spec_list = [
      'km,kn',  # C += A^T.B  fastest
      'mk,kn',  # C += A.B
      'mk,nk'  # C += A.B^T  slowest
    ]

    for line in all_sizes:
      if line[0] == '#':
        continue
      matrix_size = line.strip('\n')
      size = [int(x) for x in line.split('x')]

      for expert in expert_list:
        for spec in spec_list:
          for compile in compile_time_name_list:
            config = sandbox_config_options(size, expert, spec, compile)
            spec_name = spec.replace(',', '')
            compile_name = compile.replace(' ', '')
            output_path = Path(args.save_dir) / f'{matrix_size}_{expert}_{spec_name}_{compile_name}.json'
            with output_path.open('w') as of:
              of.write(json.dumps(config))
            print("Config file is saved to", output_path)