TILE_SIZES_TO_CHECK = [18,16,14,12,10,9,8,6,4,2]

def get_first_tile(M, N):
    [is_M_tiled, is_N_tiled] = [False, False]
    for tile in TILE_SIZES_TO_CHECK:
        if((not is_M_tiled) and (M % tile)==0):
            M_tile = tile
            is_M_tiled = True
        if((not is_N_tiled) and (N % tile)==0):
            N_tile = tile
            is_N_tiled = True
        if(is_N_tiled and is_M_tiled):
            break
    
    return (M_tile, N_tile)

def get_second_tile(M, N):
    if((M % 2) == 0):
        M_tile = M/2
    else:
        M_tile = M 

    if((N % 2) == 0):
        N_tile = N/2
    else:
        N_tile = N
    
    return (int(M_tile), int(N_tile))

def get_third_tile(K, max_of_first_tile_size):
    for tile in TILE_SIZES_TO_CHECK:
        if(tile > max_of_first_tile_size):
            continue
        if((K % tile)==0):
            return tile
    return 1

def main():
    matrix_size_file = open("./../benchmark_sizes/test_sandbox.txt", "r")
    matrix_sizes = matrix_size_file.readlines()

    tile_sizes_file = open("./sandbox_tiles.txt", "w")
    for matrix_size in matrix_sizes:
        M, N, K = matrix_size.split('x')
        M = int(M)
        N = int(N)
        K = int(K)
        first_tile_size = get_first_tile(M, N)
        second_tile_size = get_second_tile(M/first_tile_size[0], N/first_tile_size[1])
        third_tile_size = get_third_tile(K, max(first_tile_size[0],first_tile_size[1]))

        tile_sizes_file.write(str(first_tile_size[0])+','+str(first_tile_size[1])+'\n')
        tile_sizes_file.write(str(second_tile_size[0])+','+str(second_tile_size[1])+'\n')
        tile_sizes_file.write('0,0,'+str(third_tile_size)+'\n')
        # print(first_tile_size)
        # print(second_tile_size)
        # print(third_tile_size)
    matrix_size_file.close()



if __name__ == "__main__":
    main()