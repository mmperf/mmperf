!elem_type_a = type f32
!elem_type_b = type f32
!elem_type_c = type f32
!row_major_A = type tensor<${M}x${K}x!elem_type_a>
!row_major_B = type tensor<${K}x${N}x!elem_type_b>
!row_major_C = type tensor<${M}x${N}x!elem_type_c>

func @matmul(
  %a: !row_major_A {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
  %b: !row_major_B {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>},
  %c: !row_major_C {linalg.buffer_layout = affine_map<(i, j)[s0, s1] -> (i, j)>}) -> !row_major_C
{
  %v0 = constant 0.0 : !elem_type_c
  %d = linalg.fill(%v0, %c) : !elem_type_c, !row_major_C -> !row_major_C
  %e = linalg.matmul ins(%a, %b : !row_major_A, !row_major_B)
    outs(%d: !row_major_C) -> !row_major_C
  return %e : !row_major_C
}