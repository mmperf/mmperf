import os
import sys
import glob
import json
import subprocess
import argparse
from os.path import abspath, dirname
from pathlib import Path


class IREEExecutionHandler(object):
    def __init__(self, mmperf_dir, build_dir, args):
        self.mmperf_dir = mmperf_dir
        self.build_dir = build_dir
        self.matmul_generator_src = abspath(self.mmperf_dir/'matmul-iree'/'src'/'matmul_generator.cc')
        self.iree_translate = abspath(self.build_dir/'matmul-iree'/'iree'/'tools'/'iree-compile')
        self.generate_embed_data = abspath(self.build_dir/'matmul-iree'/'iree'/'build_tools'/'embed_data'/'generate_embed_data')
        self.tiling_pass = os.path.abspath(self.build_dir/'matmul-iree'/'src'/'tiling'/'add-tiling-attribute-pass')
        self.flatc = abspath(self.build_dir/"flatbuffers-install/bin/flatc")
        self.compile_options_fbs = abspath(self.mmperf_dir/"matmul-iree/src/tiling/compile_options.fbs")
        self.mlir_objs_dir = abspath(self.build_dir/'matmul-iree'/'mlir-objs')
        self.bin_dir = abspath(self.build_dir/'matmul')
        self.compile_args = args
        self.c_compiler = 'clang-11'
        self.cxx_compiler = 'clang++-11'
        if self.compile_args.target == "iree" or self.compile_args.target == "shark":
            self.device_src = abspath(self.mmperf_dir/'matmul-iree'/'src'/'device_dylib.c')
        elif self.compile_args.target == "iree-cuda" or self.compile_args.target == "shark-cuda":
            self.device_src = abspath(self.mmperf_dir/'matmul-iree'/'src'/'device_cuda.c')
        self.multi_thread = False
        if self.compile_args.multi_thread:
            self.multi_thread = True

    def apply_tiling_pass(self, file_path, filename, matmul_size_str):
        cmd = f'{self.flatc} -b {self.compile_options_fbs} {file_path}'
        subprocess.run(cmd, shell=True, check=True, cwd=self.mlir_objs_dir)
        bin_name = file_path.split('/')[-1].split('.')[0] + '.bin'

        cmd = f'{self.tiling_pass} ' \
              f'{self.mlir_objs_dir}/matmul_{matmul_size_str}.mlir ' \
              f'--compile-options {self.mlir_objs_dir}/{bin_name} '
        subprocess.run(cmd, shell=True, check=True, cwd=self.mlir_objs_dir)
        return abspath(f'{self.mlir_objs_dir}/matmul_{matmul_size_str}-tiled.mlir')

    def translate_to_vm_bytecode(self, mlir_file, filename, swizzle=None):
        if self.compile_args.target == "iree" or self.compile_args.target == "shark":
            cmd = f'{self.iree_translate} ' \
                  f'-iree-input-type=mhlo ' \
                  f'-iree-mlir-to-vm-bytecode-module ' \
                  f'-iree-hal-target-backends=llvm-cpu ' \
                  f'-iree-llvm-target-cpu-features=host ' \
                  f'-iree-llvmcpu-enable-hoist-padding ' \
                  f'-iree-llvm-link-embedded=true ' \
                  f'-iree-llvm-debug-symbols=false ' \
                  f'-iree-vm-bytecode-module-strip-source-map=true ' \
                  f'-iree-vm-emit-polyglot-zip=false ' \
                  f'-iree-hal-benchmark-dispatch-repeat-count=100 ' \
                  f'{mlir_file} -o matmul_{str(filename)}.vmfb ' \
                  f'-iree-llvmcpu-embedded-linker-path=lld'
        elif self.compile_args.target == "iree-cuda" or self.compile_args.target == "shark-cuda":
            cmd = f'{self.iree_translate} ' \
                  f'-iree-input-type=mhlo ' \
                  f'-iree-mlir-to-vm-bytecode-module ' \
                  f'-iree-hal-target-backends=cuda ' \
                  f'-iree-hal-cuda-llvm-target-arch=sm_80 ' \
                  f'-iree-hal-benchmark-dispatch-repeat-count=100 ' \
                  f'{mlir_file} -o matmul_{str(filename)}.vmfb ' \
                  f'-iree-llvmcpu-embedded-linker-path=lld '
            if swizzle != None:
                cmd += f'-iree-codegen-log-swizzle-tile={swizzle} '
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)
        return abspath(f'{self.bin_dir}/matmul_{str(filename)}.vmfb')

    def embed_data(self, vmfb_file, filename):
        name = f'matmul_{str(filename)}'
        cmd = f'{self.generate_embed_data} ' \
              f'--output_header={name}.h --output_impl={name}.c --identifier=matmul --flatten {vmfb_file}'
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)

    def create_main(self, filename, sizes):
        if len(sizes) == 3:
            M, N, K = sizes
            B = 0
        else:
            B, M, N, K = sizes
        header = f'\"matmul_{filename}.h\"'
        cmd = f'cat {self.matmul_generator_src} | ' \
               "sed 's@MATMUL_HEADER@'" + f"'{str(header)}'" + "'@g' " \
              f' > matmul_generator_{filename}.cc'
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)
        cmd = f'{self.cxx_compiler} ' \
              f'-I{self.mmperf_dir}/external/benchmark/include ' \
              f'-I{self.mmperf_dir}/external/iree/runtime/src -c matmul_generator_{filename}.cc ' \
              f'-DFILE_NAME="matmul_{str(filename)}_perf.out" ' \
              f'-DMDIM={M} -DNDIM={N} -DKDIM={K} -DBDIM={B} ' \
              f'-Wno-reorder-init-list '
        if self.compile_args.dtype == "f16":
            cmd += f'-DUSE_FP16'
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)

    def create_device(self):
        cmd = f'{self.c_compiler} ' \
              f'-I{self.mmperf_dir}/external/iree/runtime/src -c {self.device_src} '
        if self.multi_thread:
            cmd += f'-DUSE_LOCAL_TASK'
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)

    def create_matmul_static_library(self, filename):
        cmd = f'{self.c_compiler} ' \
              f'-include matmul_{str(filename)}.h -c matmul_{str(filename)}.c'
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)
        cmd = f'ar -rv matmul_{str(filename)}.a matmul_{str(filename)}.o'
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)
        cmd = f'rm matmul_{str(filename)}.o'
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)

    def create_matmul_executable(self, filename):
        iree = abspath(self.build_dir/'matmul-iree'/'iree'/'runtime'/'src'/'iree')
        third_party = abspath(self.build_dir/'matmul-iree'/'iree'/'third_party')
        build_tools = abspath(self.build_dir/'matmul-iree'/'iree'/ 'build_tools')
        if self.multi_thread:
            local_driver = f'{iree}/hal/drivers/local_task/libiree_hal_drivers_local_task_task_driver.a '
        else:
            local_driver = f'{iree}/hal/drivers/local_sync/libiree_hal_drivers_local_sync_sync_driver.a '

        if self.compile_args.target == "iree" or self.compile_args.target == "shark":
            cmd = f'{self.cxx_compiler} ' \
                  f'-o matmul_{str(filename)} matmul_generator_{filename}.o device_dylib.o ' \
                  f'matmul_{filename}.a ' \
                  f'{iree}/base/libiree_base_base.a ' \
                  f'{iree}/hal/libiree_hal_hal.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_hal.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_types.a ' \
                  f'{iree}/modules/hal/utils/libiree_modules_hal_utils_buffer_diagnostics.a ' \
                  f'{iree}/vm/libiree_vm_bytecode_module.a ' \
                  f'{iree}/hal/local/loaders/libiree_hal_local_loaders_embedded_elf_loader.a ' \
                  f'{iree}/hal/local/libiree_hal_local_local.a ' \
                  f'{iree}/hal/local/libiree_hal_local_executable_loader.a ' \
                  f'{iree}/hal/local/elf/libiree_hal_local_elf_elf_module.a ' \
                  f'{iree}/hal/local/elf/libiree_hal_local_elf_arch.a ' \
                  f'{iree}/hal/local/elf/libiree_hal_local_elf_platform.a ' \
                  f'{iree}/base/internal/libiree_base_internal_dynamic_library.a ' \
                  f'{iree}/base/internal/libiree_base_internal_path.a ' \
                  f'{iree}/base/internal/libiree_base_internal_event_pool.a ' \
                  f'{local_driver}' \
                  f'{iree}/hal/local/libiree_hal_local_local.a ' \
                  f'{iree}/hal/local/libiree_hal_local_executable_environment.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_buffer_transfer.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_hal.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_types.a ' \
                  f'{iree}/modules/hal/utils/libiree_modules_hal_utils_buffer_diagnostics.a ' \
                  f'{iree}/base/internal/libiree_base_internal_arena.a ' \
                  f'{iree}/base/internal/libiree_base_internal_fpu_state.a ' \
                  f'{iree}/base/internal/libiree_base_internal_cpu.a ' \
                  f'{iree}/task/libiree_task_api.a ' \
                  f'{iree}/base/internal/libiree_base_internal_flags.a ' \
                  f'{iree}/base/internal/libiree_base_internal_file_io.a ' \
                  f'{iree}/task/libiree_task_task.a ' \
                  f'{iree}/base/internal/libiree_base_internal_wait_handle.a ' \
                  f'{iree}/base/internal/libiree_base_internal_atomic_slist.a ' \
                  f'{third_party}/cpuinfo/libcpuinfo.a ' \
                  f'{third_party}/cpuinfo/deps/clog/libclog.a ' \
                  f'{iree}/base/internal/libiree_base_internal_threading.a ' \
                  f'{iree}/hal/libiree_hal_hal.a ' \
                  f'{iree}/base/internal/libiree_base_internal_synchronization.a ' \
                  f'{iree}/vm/libiree_vm_impl.a ' \
                  f'{iree}/base/internal/libiree_base_internal_arena.a ' \
                  f'{iree}/base/internal/libiree_base_internal_atomic_slist.a ' \
                  f'{iree}/base/internal/libiree_base_internal_dynamic_library.a ' \
                  f'{iree}/base/internal/libiree_base_internal_fpu_state.a ' \
                  f'{iree}/base/internal/libiree_base_internal_cpu.a ' \
                  f'{iree}/base/internal/libiree_base_internal_file_io.a ' \
                  f'{iree}/base/internal/libiree_base_internal_path.a ' \
                  f'{iree}/base/internal/libiree_base_internal_flags.a ' \
                  f'{iree}/base/internal/libiree_base_internal_synchronization.a ' \
                  f'{iree}/base/internal/libiree_base_internal_threading.a ' \
                  f'{iree}/base/internal/libiree_base_internal_wait_handle.a ' \
                  f'{iree}/base/internal/libiree_base_internal_event_pool.a ' \
                  f'{iree}/base/libiree_base_base.a ' \
                  f'{iree}/hal/libiree_hal_hal.a ' \
                  f'{iree}/hal/local/elf/libiree_hal_local_elf_arch.a ' \
                  f'{iree}/hal/local/elf/libiree_hal_local_elf_elf_module.a ' \
                  f'{iree}/hal/local/elf/libiree_hal_local_elf_platform.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_buffer_transfer.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_semaphore_base.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_resource_set.a ' \
                  f'{iree}/hal/local/libiree_hal_local_local.a ' \
                  f'{local_driver}' \
                  f'{iree}/hal/local/libiree_hal_local_executable_environment.a ' \
                  f'{iree}/hal/local/loaders/libiree_hal_local_loaders_embedded_elf_loader.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_hal.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_types.a ' \
                  f'{iree}/task/libiree_task_api.a ' \
                  f'{iree}/task/libiree_task_task.a ' \
                  f'{iree}/vm/libiree_vm_bytecode_module.a ' \
                  f'{iree}/vm/libiree_vm_impl.a ' \
                  f'{third_party}/cpuinfo/deps/clog/libclog.a ' \
                  f'{third_party}/cpuinfo/libcpuinfo.a ' \
                  f'{build_tools}/third_party/flatcc/libflatcc_runtime.a ' \
                  f'{build_tools}/third_party/flatcc/libflatcc_parsing.a ' \
                  f'{third_party}/llvm-project/llvm/lib/libLLVMAnalysis.a ' \
                  f'{self.build_dir}/benchmark/src/benchmark-build/lib/libgmock.a ' \
                  f'{self.build_dir}/benchmark/src/benchmark-build/src/libbenchmark.a ' \
                  f'-lpthread ' \
                  f'-ldl ' \
                  f'-lm '
        elif self.compile_args.target == "iree-cuda" or self.compile_args.target == "shark-cuda":
            cmd = f'{self.cxx_compiler} ' \
                  f'-o matmul_{str(filename)} matmul_generator_{filename}.o device_cuda.o ' \
                  f'matmul_{filename}.a ' \
                  f'{iree}/base/libiree_base_base.a ' \
                  f'{iree}/hal/libiree_hal_hal.a ' \
                  f'{iree}/hal/drivers/cuda/registration/libiree_hal_drivers_cuda_registration_registration.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_hal.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_types.a ' \
                  f'{iree}/modules/hal/utils/libiree_modules_hal_utils_buffer_diagnostics.a ' \
                  f'{iree}/vm/libiree_vm_bytecode_module.a ' \
                  f'{iree}/base/internal/libiree_base_internal_dynamic_library.a ' \
                  f'{iree}/base/internal/libiree_base_internal_path.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_hal.a ' \
                  f'{iree}/base/internal/libiree_base_internal_arena.a ' \
                  f'{iree}/base/internal/libiree_base_internal_fpu_state.a ' \
                  f'{iree}/task/libiree_task_api.a ' \
                  f'{iree}/base/internal/libiree_base_internal_flags.a ' \
                  f'{iree}/base/internal/libiree_base_internal_file_io.a ' \
                  f'{iree}/task/libiree_task_task.a ' \
                  f'{iree}/base/internal/libiree_base_internal_wait_handle.a ' \
                  f'{iree}/base/internal/libiree_base_internal_atomic_slist.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_buffer_transfer.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_resource_set.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_deferred_command_buffer.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_semaphore_base.a ' \
                  f'{iree}/hal/drivers/cuda/libiree_hal_drivers_cuda_cuda.a ' \
                  f'{iree}/hal/drivers/cuda/libiree_hal_drivers_cuda_dynamic_symbols.a ' \
                  f'{iree}/base/internal/libiree_base_internal_threading.a ' \
                  f'{iree}/hal/libiree_hal_hal.a ' \
                  f'{iree}/base/internal/libiree_base_internal_synchronization.a ' \
                  f'{iree}/vm/libiree_vm_impl.a ' \
                  f'{iree}/base/internal/libiree_base_internal_arena.a ' \
                  f'{iree}/base/internal/libiree_base_internal_atomic_slist.a ' \
                  f'{iree}/base/internal/libiree_base_internal_dynamic_library.a ' \
                  f'{iree}/base/internal/libiree_base_internal_fpu_state.a ' \
                  f'{iree}/base/internal/libiree_base_internal_file_io.a ' \
                  f'{iree}/base/internal/libiree_base_internal_path.a ' \
                  f'{iree}/base/internal/libiree_base_internal_flags.a ' \
                  f'{iree}/base/internal/libiree_base_internal_synchronization.a ' \
                  f'{iree}/base/internal/libiree_base_internal_threading.a ' \
                  f'{iree}/base/internal/libiree_base_internal_wait_handle.a ' \
                  f'{iree}/base/libiree_base_base.a ' \
                  f'{iree}/hal/drivers/cuda/registration/libiree_hal_drivers_cuda_registration_registration.a ' \
                  f'{iree}/hal/libiree_hal_hal.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_hal.a ' \
                  f'{iree}/modules/hal/libiree_modules_hal_types.a ' \
                  f'{iree}/modules/hal/utils/libiree_modules_hal_utils_buffer_diagnostics.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_deferred_command_buffer.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_buffer_transfer.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_resource_set.a ' \
                  f'{iree}/hal/utils/libiree_hal_utils_semaphore_base.a ' \
                  f'{iree}/hal/drivers/cuda/libiree_hal_drivers_cuda_cuda.a ' \
                  f'{iree}/hal/drivers/cuda/libiree_hal_drivers_cuda_dynamic_symbols.a ' \
                  f'{iree}/task/libiree_task_api.a ' \
                  f'{iree}/task/libiree_task_task.a ' \
                  f'{iree}/vm/libiree_vm_bytecode_module.a ' \
                  f'{iree}/vm/libiree_vm_impl.a ' \
                  f'{third_party}/cpuinfo/deps/clog/libclog.a ' \
                  f'{third_party}/cpuinfo/libcpuinfo.a ' \
                  f'{build_tools}/third_party/flatcc/libflatcc_runtime.a ' \
                  f'{build_tools}/third_party/flatcc/libflatcc_parsing.a ' \
                  f'{third_party}/llvm-project/llvm/lib/libLLVMAnalysis.a ' \
                  f'{self.build_dir}/benchmark/src/benchmark-build/lib/libgmock.a ' \
                  f'{self.build_dir}/benchmark/src/benchmark-build/src/libbenchmark.a ' \
                  f'-lpthread ' \
                  f'-ldl ' \
                  f'-lm '
        subprocess.run(cmd, shell=True, check=True, cwd=self.bin_dir)

    def generate_nodai_bins(self, f_path, filename, matmul_size, swizzle=None):
        self.create_device()
        matmul_size_str = 'x'.join([str(d) for d in tuple(matmul_size)])
        print(f'Compiling {self.mlir_objs_dir}/matmul_{matmul_size_str}.mlir',
              f'for matmul size {int(matmul_size[0])}x{int(matmul_size[1])}x{int(matmul_size[2])}')
        mlir_tiled_file = self.apply_tiling_pass(f_path, filename, matmul_size_str)
        vmfb_file = self.translate_to_vm_bytecode(mlir_tiled_file, filename, swizzle)
        self.embed_data(vmfb_file, filename)
        self.create_main(filename, matmul_size)
        self.create_matmul_static_library(filename)
        self.create_matmul_executable(filename)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', default="", help='Path to load config files')
    parser.add_argument('-mmperf_build', default="", help='Path to mmperf build dir.')
    parser.add_argument('-mmperf_src', default="../../../thirdparty/mmperf", help='Path to mmperf source dir.')
    parser.add_argument('-dtype', default='f32', help='Data precision fp32 or fp16')
    args = parser.parse_args(argv[1:])
    args.target = 'shark-cuda'

    mmperf_src = Path(args.mmperf_src)
    mmperf_build = Path(dirname(abspath(__file__))) / args.mmperf_build
    exec_handle = IREEExecutionHandler(mmperf_src, mmperf_build, args)
    for f_path in glob.glob(abspath(os.path.join(args.config_path, '*.json'))):
        with open(f_path, 'r') as f:
            data = json.load(f)
            if "b" in data.keys() and data["b"] != 0:
              matmul_size = [int(data["b"]), int(data["m"]), int(data["n"]), int(data["k"])]
            else:
              matmul_size = [int(data["m"]), int(data["n"]), int(data["k"])]

            best_config = data
            try:
                best_swizzle = data["options"][0]["swizzle"]
            except:
                best_swizzle = None
            print("Best config", best_config)

            matmul_size_str = 'x'.join([str(d) for d in tuple(matmul_size)])
            file_name = f'nodai-shark-cuda_{matmul_size_str}'
            exec_handle.generate_nodai_bins(f_path, file_name, matmul_size, swizzle=best_swizzle)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
