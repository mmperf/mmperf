// This is a copy of matmul_f16_base.mlir from (https://github.com/google/iree-llvm-sandbox/blob/main/runners/test/matmul_f16_base.mlir)
!elem_type_a = type f16
!elem_type_b = type f16
!elem_type_c = type f16
!row_major_A = type tensor<${M}x${K}x!elem_type_a>
!row_major_B = type tensor<${K}x${N}x!elem_type_b>
!row_major_C = type tensor<${M}x${N}x!elem_type_c>

func @init_and_matmul(
  %a:
    !row_major_A {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
  %b:
    !row_major_B {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
  %c:
    !row_major_C {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>})
  -> !row_major_C
{
  %v0 = constant 0.0 : !elem_type_c
  %d = linalg.fill(%v0, %c) : !elem_type_c, !row_major_C -> !row_major_C
  %e = linalg.matmul ins(%a, %b : !row_major_A, !row_major_B)
    outs(%d: !row_major_C) -> !row_major_C
  return %e : !row_major_C
}

func @print_perf(%iters: index, %total_time: f64) {
  %c2 = constant 2 : index
  %cM = constant ${M} : index
  %cN = constant ${N} : index
  %cK = constant ${K} : index

  %mn = muli %cM, %cN : index
  %mnk = muli %mn, %cK : index

  // 2*M*N*K.
  %flops_per_iter = muli %c2, %mnk : index
  %flops = muli %iters, %flops_per_iter : index
  %flops_i64 = index_cast %flops : index to i64
  %flops_f = sitofp %flops_i64 : i64 to f64
  %flops_per_s = divf %flops_f, %total_time : f64
  vector.print %flops_per_s : f64

  return
}

func @exec(%iters : index) {
  %v0 = constant 0.0 : !elem_type_c
  %v1 = constant 1.0 : !elem_type_a
  %v2 = constant 2.0 : !elem_type_b

  %c0 = constant 0: index
  %c1 = constant 1: index
  %cM = constant ${M} : index
  %cN = constant ${N} : index
  %cK = constant ${K} : index

  %A = linalg.init_tensor [${M}, ${K}] : !row_major_A
  %B = linalg.init_tensor [${K}, ${N}] : !row_major_B
  %C = linalg.init_tensor [${M}, ${N}] : !row_major_C
  %AA = linalg.fill(%v1, %A) : !elem_type_a, !row_major_A -> !row_major_A
  %BB = linalg.fill(%v2, %B) : !elem_type_b, !row_major_B -> !row_major_B
  %CC = linalg.fill(%v0, %C) : !elem_type_c, !row_major_C -> !row_major_C


  /// Run and dump performance for matmul.
  %t_start_matmul = call @rtclock() : () -> f64
  %res = scf.for %arg0 = %c0 to %iters step %c1 iter_args(%dummy = %CC) -> (!row_major_C) {
    %r = call @init_and_matmul(%AA, %BB, %dummy) : (!row_major_A, !row_major_B, !row_major_C) -> (!row_major_C)
    scf.yield %r : !row_major_C
  }
  %t_end_matmul = call @rtclock() : () -> f64
  %tmatmul = subf %t_end_matmul, %t_start_matmul: f64
  call @print_perf(%iters, %tmatmul) : (index, f64) -> ()

  %val = tensor.extract %res[%c0, %c0]: !row_major_C
  %valf32 = fpext %val : f16 to f32
  %v00 = constant dense<0.0> : vector<1xf32>
  %vec = vector.insert %valf32, %v00[0] : f32 into vector<1xf32>
  vector.print %vec: vector<1xf32>

  return
}

func @main() {
  %iters = constant ${ITERS} : index
  call @exec(%iters) : (index) -> ()
  return
}

func private @rtclock() -> f64
