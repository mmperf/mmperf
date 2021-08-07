# Debugging matrix multiplication in MLIR

This document outlines the steps I took trying to efficiently multiply
two 1024 x 1024 row-major matrices (C = A x B) in MLIR. We start with the matrices already as memrefs and the computation graph expressed in the linalg dialect shown below.

    func @matmul(%a: memref<1024x1024xf32>, %b: memref<1024x1024xf32>, %c: memref<1024x1024xf32>) {
     %cst = constant 0.000000e+00 : f32
     linalg.fill(%c, %cst) : memref<1024x1024xf32>, f32
     linalg.matmul ins(%a, %b : memref<1024x1024xf32>, memref<1024x1024xf32>)
                outs(%c: memref<1024x1024xf32>)
     return
    }

The fill operation is required because we are just computing the product of A and B. 

The next step is to apply a first level of tiling and permutation. So here we use the following codegen strategy.

    CodegenStrategy strategy;
    strategy
    .tile<MatMulOp>(LinalgTilingOptions().setTileSizes({128, 128, 256}).setInterchange({0, 2, 1}))
    .promote<MatMulOp>(LinalgPromotionOptions().setOperandsToPromote({0, 1})
      .setAlignment(getpagesize()));

This results in the following IR.

    #map0 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
    #map1 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
    #map2 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
    module  {
        func @matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) attributes {passthrough = [["prefer-vector-width", "256"], ["target-cpu", "haswell"]]} {
            %cst = constant 0.000000e+00 : f32
            %c1024 = constant 1024 : index
            %c256 = constant 256 : index
            %c128 = constant 128 : index
            %c131072 = constant 131072 : index
            %c0 = constant 0 : index
            linalg.fill(%arg2, %cst) : memref<1024x1024xf32>, f32
            %0 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            %1 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            scf.for %arg3 = %c0 to %c1024 step %c128 {
                scf.for %arg4 = %c0 to %c1024 step %c256 {
                    %2 = subview %arg0[%arg3, %arg4] [128, 256] [1, 1] : memref<1024x1024xf32> to memref<128x256xf32, #map0>
                    scf.for %arg5 = %c0 to %c1024 step %c128 {
                        %3 = subview %arg1[%arg4, %arg5] [256, 128] [1, 1] : memref<1024x1024xf32> to memref<256x128xf32, #map0>
                        %4 = subview %arg2[%arg3, %arg5] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, #map0>
                        %5 = std.view %0[%c0][] : memref<?xi8> to memref<128x256xf32>
                        %6 = subview %5[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, #map1>
                        %7 = std.view %1[%c0][] : memref<?xi8> to memref<256x128xf32>
                        %8 = subview %7[0, 0] [256, 128] [1, 1] : memref<256x128xf32> to memref<256x128xf32, #map2>
                        linalg.copy(%2, %6) : memref<128x256xf32, #map0>, memref<128x256xf32, #map1>
                        linalg.copy(%3, %8) : memref<256x128xf32, #map0>, memref<256x128xf32, #map2>
                        linalg.matmul ins(%6, %8 : memref<128x256xf32, #map1>, memref<256x128xf32, #map2>) outs(%4 : memref<128x128xf32, #map0>)
                    }
                }
            }
            dealloc %1 : memref<?xi8>
            dealloc %0 : memref<?xi8>
            return
        }
    }

Using LICM, we can hoist a lot of the ops out of the innermost loops. Doing so gives us the following

    #map0 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
    #map1 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
    #map2 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
    module  {
        func @matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) attributes {passthrough = [["prefer-vector-width", "256"], ["target-cpu", "haswell"]]} {
            %cst = constant 0.000000e+00 : f32
            %c1024 = constant 1024 : index
            %c256 = constant 256 : index
            %c128 = constant 128 : index
            %c131072 = constant 131072 : index
            %c0 = constant 0 : index
            linalg.fill(%arg2, %cst) : memref<1024x1024xf32>, f32
            %0 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            %1 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            %5 = std.view %0[%c0][] : memref<?xi8> to memref<128x256xf32>
            %6 = subview %5[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, #map1>
            %7 = std.view %1[%c0][] : memref<?xi8> to memref<256x128xf32>
            %8 = subview %7[0, 0] [256, 128] [1, 1] : memref<256x128xf32> to memref<256x128xf32, #map2>
            scf.for %arg3 = %c0 to %c1024 step %c128 {
                scf.for %arg4 = %c0 to %c1024 step %c256 {
                    %2 = subview %arg0[%arg3, %arg4] [128, 256] [1, 1] : memref<1024x1024xf32> to memref<128x256xf32, #map0>
                    linalg.copy(%2, %6) : memref<128x256xf32, #map0>, memref<128x256xf32, #map1>
                    scf.for %arg5 = %c0 to %c1024 step %c128 {
                        %3 = subview %arg1[%arg4, %arg5] [256, 128] [1, 1] : memref<1024x1024xf32> to memref<256x128xf32, #map0>
                        %4 = subview %arg2[%arg3, %arg5] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, #map0>
                        linalg.copy(%3, %8) : memref<256x128xf32, #map0>, memref<256x128xf32, #map2>
                        linalg.matmul ins(%6, %8 : memref<128x256xf32, #map1>, memref<256x128xf32, #map2>) outs(%4 : memref<128x128xf32, #map0>)
                    }
                }
            }
            dealloc %1 : memref<?xi8>
            dealloc %0 : memref<?xi8>
            return
        }
    }

To reduce register pressure for subsequent register tiling, we can outline the matmul op as shown below.

    #map0 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
    #map1 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
    #map2 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
    module  {
        func @matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) attributes {passthrough = [["prefer-vector-width", "256"], ["target-cpu", "haswell"]]} {
            %cst = constant 0.000000e+00 : f32
            %c1024 = constant 1024 : index
            %c256 = constant 256 : index
            %c128 = constant 128 : index
            %c131072 = constant 131072 : index
            %c0 = constant 0 : index
            linalg.fill(%arg2, %cst) : memref<1024x1024xf32>, f32
            %0 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            %1 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            %5 = std.view %0[%c0][] : memref<?xi8> to memref<128x256xf32>
            %6 = subview %5[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, #map1>
            %7 = std.view %1[%c0][] : memref<?xi8> to memref<256x128xf32>
            %8 = subview %7[0, 0] [256, 128] [1, 1] : memref<256x128xf32> to memref<256x128xf32, #map2>
            scf.for %arg3 = %c0 to %c1024 step %c128 {
                scf.for %arg4 = %c0 to %c1024 step %c256 {
                    %2 = subview %arg0[%arg3, %arg4] [128, 256] [1, 1] : memref<1024x1024xf32> to memref<128x256xf32, #map0>
                    linalg.copy(%2, %6) : memref<128x256xf32, #map0>, memref<128x256xf32, #map1>
                    scf.for %arg5 = %c0 to %c1024 step %c128 {
                        %3 = subview %arg1[%arg4, %arg5] [256, 128] [1, 1] : memref<1024x1024xf32> to memref<256x128xf32, #map0>
                        %4 = subview %arg2[%arg3, %arg5] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, #map0>
                        linalg.copy(%3, %8) : memref<256x128xf32, #map0>, memref<256x128xf32, #map2>
                        call @outlined_matmul(%6, %8, %4) : (memref<128x256xf32, #map1>, memref<256x128xf32, #map2>, memref<128x128xf32, #map0>) -> ()
                    }
                }
            }
            dealloc %1 : memref<?xi8>
            dealloc %0 : memref<?xi8>
            return
        }
        func @outlined_matmul(%arg0: memref<128x256xf32, #map1>, %arg1: memref<256x128xf32, #map2>, %arg2: memref<128x128xf32, #map0>) {
            linalg.matmul ins(%arg0, %arg1 : memref<128x256xf32, #map1>, memref<256x128xf32, #map2>) outs(%arg2 : memref<128x128xf32, #map0>)
            return
        }
    }

Another thing we can do is loop fusion and bring in the zeroing of the C matrix closer to the rest of the work. The result of applying this transform is shown below.

    #map0 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
    #map1 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
    #map2 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
    module  {
        func @matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) attributes {passthrough = [["prefer-vector-width", "256"], ["target-cpu", "haswell"]]} {
            %cst = constant 0.000000e+00 : f32
            %c1024 = constant 1024 : index
            %c256 = constant 256 : index
            %c128 = constant 128 : index
            %c131072 = constant 131072 : index
            %c0 = constant 0 : index
            %0 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            %1 = alloc(%c131072) {alignment = 4096 : i64} : memref<?xi8>
            %5 = std.view %0[%c0][] : memref<?xi8> to memref<128x256xf32>
            %6 = subview %5[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, #map1>
            %7 = std.view %1[%c0][] : memref<?xi8> to memref<256x128xf32>
            %8 = subview %7[0, 0] [256, 128] [1, 1] : memref<256x128xf32> to memref<256x128xf32, #map2>
            scf.for %arg3 = %c0 to %c1024 step %c128 {
                scf.for %arg5 = %c0 to %c1024 step %c128 {
                    %4 = subview %arg2[%arg3, %arg5] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, #map0>
                    linalg.fill(%4, %cst) : memref<128x128xf32, #map0>, f32
                }
                scf.for %arg4 = %c0 to %c1024 step %c256 {
                    %2 = subview %arg0[%arg3, %arg4] [128, 256] [1, 1] : memref<1024x1024xf32> to memref<128x256xf32, #map0>
                    linalg.copy(%2, %6) : memref<128x256xf32, #map0>, memref<128x256xf32, #map1>
                    scf.for %arg5 = %c0 to %c1024 step %c128 {
                        %3 = subview %arg1[%arg4, %arg5] [256, 128] [1, 1] : memref<1024x1024xf32> to memref<256x128xf32, #map0>
                        %4 = subview %arg2[%arg3, %arg5] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, #map0>
                        linalg.copy(%3, %8) : memref<256x128xf32, #map0>, memref<256x128xf32, #map2>
                        call @outlined_matmul(%6, %8, %4) : (memref<128x256xf32, #map1>, memref<256x128xf32, #map2>, memref<128x128xf32, #map0>) -> ()
                    }
                }
            }
            dealloc %1 : memref<?xi8>
            dealloc %0 : memref<?xi8>
            return
        }
        func @outlined_matmul(%arg0: memref<128x256xf32, #map1>, %arg1: memref<256x128xf32, #map2>, %arg2: memref<128x128xf32, #map0>) {
            linalg.matmul ins(%arg0, %arg1 : memref<128x256xf32, #map1>, memref<256x128xf32, #map2>) outs(%arg2 : memref<128x128xf32, #map0>)
            return
        }
    }

Having done this, we can now apply 2d vectorization on the non-outlines copies and register level tiling on the outlined matmul.
    
    // Step 2: a simple 2-D copy vectorization on the non-outlined copies.
    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<FillOp>(LinalgTilingOptions().setTileSizes({4, 16}))
        .vectorize<FillOp>()
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }

    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<CopyOp>(LinalgTilingOptions().setTileSizes({4, 16}))
        .vectorize<CopyOp>()
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }

    // Step 3: apply the register level strategy on the outlined piece.
    {
      CodegenStrategy strategyRegisters;
      strategyRegisters
        .tile<MatmulOp>(LinalgTilingOptions().setTileSizes({8, 16, 8}))
        .promote<MatmulOp>(LinalgPromotionOptions()
                             .setUseFullTileBuffersByDefault(registerPromoteFullTile)
                             .setAlignment(128))
        .vectorize<MatmulOp>()
        .setVectorTransformsOptions(
            vector::VectorTransformsOptions()
                .setVectorTransformsOptions(vectorContractLowering)
                .setVectorTransferSplit(vectorTransferSplit))
        .setVectorTransferToSCFOptions(
            VectorTransferToSCFOptions().setUnroll(unrollVectorTransfers));

      strategyRegisters.transform(getFunction());
    }
    
    
    
This results in the following IR.

    #map0 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
    #map1 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
    #map2 = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
    #map3 = affine_map<(d0, d1)[s0] -> (d0 * 256 + s0 + d1)>
    #map4 = affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>
    module  {
        func @matmul(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) attributes {passthrough = [["prefer-vector-width", "256"], ["target-cpu", "haswell"]]} {
            %c1024 = constant 1024 : index
            %cst = constant dense<0.000000e+00> : vector<4x16xf32>
            %c4 = constant 4 : index
            %c256 = constant 256 : index
            %c16 = constant 16 : index
            %c128 = constant 128 : index
            %cst_0 = constant 0.000000e+00 : f32
            %c0 = constant 0 : index
            %c1 = constant 1 : index
            %c2 = constant 2 : index
            %c3 = constant 3 : index
            %0 = alloc() : memref<131072xi8>
            %1 = alloc() : memref<131072xi8>
            %2 = std.view %0[%c0][] : memref<131072xi8> to memref<128x256xf32>
            %3 = subview %2[0, 0] [128, 256] [1, 1] : memref<128x256xf32> to memref<128x256xf32, #map0>
            %4 = std.view %1[%c0][] : memref<131072xi8> to memref<256x128xf32>
            %5 = subview %4[0, 0] [256, 128] [1, 1] : memref<256x128xf32> to memref<256x128xf32, #map1>
            %6 = vector.extract %cst[0] : vector<4x16xf32>
            %7 = vector.extract %cst[1] : vector<4x16xf32>
            %8 = vector.extract %cst[2] : vector<4x16xf32>
            %9 = vector.extract %cst[3] : vector<4x16xf32>
            scf.for %arg3 = %c0 to %c1024 step %c128 {
                scf.for %arg4 = %c0 to %c1024 step %c128 {
                    %10 = subview %arg2[%arg3, %arg4] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, #map2>
                    scf.for %arg5 = %c0 to %c128 step %c4 {
                        scf.for %arg6 = %c0 to %c128 step %c16 {
                            %11 = subview %10[%arg5, %arg6] [4, 16] [1, 1] : memref<128x128xf32, #map2> to memref<4x16xf32, #map2>
                            vector.transfer_write %6, %11[%c0, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map2>
                            vector.transfer_write %7, %11[%c1, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map2>
                            vector.transfer_write %8, %11[%c2, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map2>
                            vector.transfer_write %9, %11[%c3, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map2>
                        }
                    }
                }
            scf.for %arg4 = %c0 to %c1024 step %c256 {
                %10 = subview %arg0[%arg3, %arg4] [128, 256] [1, 1] : memref<1024x1024xf32> to memref<128x256xf32, #map2>
                scf.for %arg5 = %c0 to %c128 step %c4 {
                    scf.for %arg6 = %c0 to %c256 step %c16 {
                        %11 = subview %10[%arg5, %arg6] [4, 16] [1, 1] : memref<128x256xf32, #map2> to memref<4x16xf32, #map2>
                        %12 = subview %3[%arg5, %arg6] [4, 16] [1, 1] : memref<128x256xf32, #map0> to memref<4x16xf32, #map3>
                        %13 = vector.transfer_read %11[%c0, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                        %14 = vector.transfer_read %11[%c1, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                        %15 = vector.transfer_read %11[%c2, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                        %16 = vector.transfer_read %11[%c3, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                        vector.transfer_write %13, %12[%c0, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map3>
                        vector.transfer_write %14, %12[%c1, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map3>
                        vector.transfer_write %15, %12[%c2, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map3>
                        vector.transfer_write %16, %12[%c3, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map3>
                    }
                }
                scf.for %arg5 = %c0 to %c1024 step %c128 {
                    %11 = subview %arg1[%arg4, %arg5] [256, 128] [1, 1] : memref<1024x1024xf32> to memref<256x128xf32, #map2>
                    %12 = subview %arg2[%arg3, %arg5] [128, 128] [1, 1] : memref<1024x1024xf32> to memref<128x128xf32, #map2>
                    scf.for %arg6 = %c0 to %c256 step %c4 {
                        scf.for %arg7 = %c0 to %c128 step %c16 {
                            %13 = subview %11[%arg6, %arg7] [4, 16] [1, 1] : memref<256x128xf32, #map2> to memref<4x16xf32, #map2>
                            %14 = subview %5[%arg6, %arg7] [4, 16] [1, 1] : memref<256x128xf32, #map1> to memref<4x16xf32, #map4>
                            %15 = vector.transfer_read %13[%c0, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                            %16 = vector.transfer_read %13[%c1, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                            %17 = vector.transfer_read %13[%c2, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                            %18 = vector.transfer_read %13[%c3, %c0], %cst_0 {masked = [false]} : memref<4x16xf32, #map2>, vector<16xf32>
                            vector.transfer_write %15, %14[%c0, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map4>
                            vector.transfer_write %16, %14[%c1, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map4>
                            vector.transfer_write %17, %14[%c2, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map4>
                            vector.transfer_write %18, %14[%c3, %c0] {masked = [false]} : vector<16xf32>, memref<4x16xf32, #map4>
                        }
                    }
                    call @outlined_matmul(%3, %5, %12) : (memref<128x256xf32, #map0>, memref<256x128xf32, #map1>, memref<128x128xf32, #map2>) -> ()
                    }
                }
            }
            dealloc %1 : memref<131072xi8>
            dealloc %0 : memref<131072xi8>
            return
        }
        func @outlined_matmul(%arg0: memref<128x256xf32, #map0>, %arg1: memref<256x128xf32, #map1>, %arg2: memref<128x128xf32, #map2>) attributes {passthrough = [["prefer-vector-width", "256"], ["target-cpu", "haswell"]]} {
            %c128 = constant 128 : index
            %c256 = constant 256 : index
            %c8 = constant 8 : index
            %c16 = constant 16 : index
            %c512 = constant 512 : index
            %cst = constant 0.000000e+00 : f32
            %cst_0 = constant dense<0.000000e+00> : vector<8x16xf32>
            %cst_1 = constant dense<0.000000e+00> : vector<8x8xf32>
            %c0 = constant 0 : index
            %c1 = constant 1 : index
            %c2 = constant 2 : index
            %c3 = constant 3 : index
            %c4 = constant 4 : index
            %c5 = constant 5 : index
            %c6 = constant 6 : index
            %c7 = constant 7 : index
            %0 = alloc(%c256) {alignment = 128 : i64} : memref<?xi8>
            %1 = alloc(%c512) {alignment = 128 : i64} : memref<?xi8>
            %2 = alloc(%c512) {alignment = 128 : i64} : memref<?xi8>
            scf.for %arg3 = %c0 to %c128 step %c8 {
                scf.for %arg4 = %c0 to %c128 step %c16 {
                    %3 = subview %arg2[%arg3, %arg4] [8, 16] [1, 1] : memref<128x128xf32, #map2> to memref<8x16xf32, #map2>
                    %4 = vector.transfer_read %3[%c0, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %5 = vector.insert %4, %cst_0 [0] : vector<16xf32> into vector<8x16xf32>
                    %6 = vector.transfer_read %3[%c1, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %7 = vector.insert %6, %5 [1] : vector<16xf32> into vector<8x16xf32>
                    %8 = vector.transfer_read %3[%c2, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %9 = vector.insert %8, %7 [2] : vector<16xf32> into vector<8x16xf32>
                    %10 = vector.transfer_read %3[%c3, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %11 = vector.insert %10, %9 [3] : vector<16xf32> into vector<8x16xf32>
                    %12 = vector.transfer_read %3[%c4, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %13 = vector.insert %12, %11 [4] : vector<16xf32> into vector<8x16xf32>
                    %14 = vector.transfer_read %3[%c5, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %15 = vector.insert %14, %13 [5] : vector<16xf32> into vector<8x16xf32>
                    %16 = vector.transfer_read %3[%c6, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %17 = vector.insert %16, %15 [6] : vector<16xf32> into vector<8x16xf32>
                    %18 = vector.transfer_read %3[%c7, %c0], %cst {masked = [false]} : memref<8x16xf32, #map2>, vector<16xf32>
                    %19 = vector.insert %18, %17 [7] : vector<16xf32> into vector<8x16xf32>
                    %20 = scf.for %arg5 = %c0 to %c256 step %c8 iter_args(%arg6 = %19) -> (vector<8x16xf32>) {
                    %29 = subview %arg0[%arg3, %arg5] [8, 8] [1, 1] : memref<128x256xf32, #map0> to memref<8x8xf32, #map3>
                    %30 = subview %arg1[%arg5, %arg4] [8, 16] [1, 1] : memref<256x128xf32, #map1> to memref<8x16xf32, #map4>
                    %31 = vector.transfer_read %29[%c0, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %32 = vector.insert %31, %cst_1 [0] : vector<8xf32> into vector<8x8xf32>
                    %33 = vector.transfer_read %29[%c1, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %34 = vector.insert %33, %32 [1] : vector<8xf32> into vector<8x8xf32>
                    %35 = vector.transfer_read %29[%c2, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %36 = vector.insert %35, %34 [2] : vector<8xf32> into vector<8x8xf32>
                    %37 = vector.transfer_read %29[%c3, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %38 = vector.insert %37, %36 [3] : vector<8xf32> into vector<8x8xf32>
                    %39 = vector.transfer_read %29[%c4, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %40 = vector.insert %39, %38 [4] : vector<8xf32> into vector<8x8xf32>
                    %41 = vector.transfer_read %29[%c5, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %42 = vector.insert %41, %40 [5] : vector<8xf32> into vector<8x8xf32>
                    %43 = vector.transfer_read %29[%c6, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %44 = vector.insert %43, %42 [6] : vector<8xf32> into vector<8x8xf32>
                    %45 = vector.transfer_read %29[%c7, %c0], %cst {masked = [false]} : memref<8x8xf32, #map3>, vector<8xf32>
                    %46 = vector.insert %45, %44 [7] : vector<8xf32> into vector<8x8xf32>
                    %47 = vector.transfer_read %30[%c0, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %48 = vector.transfer_read %30[%c1, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %49 = vector.transfer_read %30[%c2, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %50 = vector.transfer_read %30[%c3, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %51 = vector.transfer_read %30[%c4, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %52 = vector.transfer_read %30[%c5, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %53 = vector.transfer_read %30[%c6, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %54 = vector.transfer_read %30[%c7, %c0], %cst {masked = [false]} : memref<8x16xf32, #map4>, vector<16xf32>
                    %55 = vector.transpose %46, [1, 0] : vector<8x8xf32> to vector<8x8xf32>
                    %56 = vector.extract %55[0] : vector<8x8xf32>
                    %57 = vector.outerproduct %56, %47, %arg6 : vector<8xf32>, vector<16xf32>
                    %58 = vector.extract %55[1] : vector<8x8xf32>
                    %59 = vector.outerproduct %58, %48, %57 : vector<8xf32>, vector<16xf32>
                    %60 = vector.extract %55[2] : vector<8x8xf32>
                    %61 = vector.outerproduct %60, %49, %59 : vector<8xf32>, vector<16xf32>
                    %62 = vector.extract %55[3] : vector<8x8xf32>
                    %63 = vector.outerproduct %62, %50, %61 : vector<8xf32>, vector<16xf32>
                    %64 = vector.extract %55[4] : vector<8x8xf32>
                    %65 = vector.outerproduct %64, %51, %63 : vector<8xf32>, vector<16xf32>
                    %66 = vector.extract %55[5] : vector<8x8xf32>
                    %67 = vector.outerproduct %66, %52, %65 : vector<8xf32>, vector<16xf32>
                    %68 = vector.extract %55[6] : vector<8x8xf32>
                    %69 = vector.outerproduct %68, %53, %67 : vector<8xf32>, vector<16xf32>
                    %70 = vector.extract %55[7] : vector<8x8xf32>
                    %71 = vector.outerproduct %70, %54, %69 : vector<8xf32>, vector<16xf32>
                    scf.yield %71 : vector<8x16xf32>
                    }
                    %21 = vector.extract %20[0] : vector<8x16xf32>
                    vector.transfer_write %21, %3[%c0, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                    %22 = vector.extract %20[1] : vector<8x16xf32>
                    vector.transfer_write %22, %3[%c1, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                    %23 = vector.extract %20[2] : vector<8x16xf32>
                    vector.transfer_write %23, %3[%c2, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                    %24 = vector.extract %20[3] : vector<8x16xf32>
                    vector.transfer_write %24, %3[%c3, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                    %25 = vector.extract %20[4] : vector<8x16xf32>
                    vector.transfer_write %25, %3[%c4, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                    %26 = vector.extract %20[5] : vector<8x16xf32>
                    vector.transfer_write %26, %3[%c5, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                    %27 = vector.extract %20[6] : vector<8x16xf32>
                    vector.transfer_write %27, %3[%c6, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                    %28 = vector.extract %20[7] : vector<8x16xf32>
                    vector.transfer_write %28, %3[%c7, %c0] {masked = [false]} : vector<16xf32>, memref<8x16xf32, #map2>
                }
            }
            dealloc %2 : memref<?xi8>
            dealloc %1 : memref<?xi8>
            dealloc %0 : memref<?xi8>
            return
        }
    }

However, despite these optimizations, we are still far from the machine's peak performance.
