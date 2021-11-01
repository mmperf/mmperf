!elem_type_a = type f32
!elem_type_b = type f32
!elem_type_c = type f32
!row_major_A = type tensor<${M}x${K}x!elem_type_a>
!row_major_B = type tensor<${K}x${N}x!elem_type_b>
!row_major_C = type tensor<${M}x${N}x!elem_type_c>

func @matmul(
  %a: !row_major_A {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
  %b: !row_major_B {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
  %c: !row_major_C {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>})
{
  %v0 = arith.constant 0.0 : !elem_type_c
  %d = linalg.fill(%v0, %c) : !elem_type_c, !row_major_C -> !row_major_C
  %e = linalg.matmul ins(%a, %b : !row_major_A, !row_major_B)
    outs(%d: !row_major_C) -> !row_major_C
  %unranked = tensor.cast %e : !row_major_C to tensor<*xf32>
  call @print_memref_f32(%unranked) : (tensor<*xf32>) -> ()

  return
}
func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
