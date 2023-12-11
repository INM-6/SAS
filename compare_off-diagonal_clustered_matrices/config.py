import itertools

N = 100

bkgr_mean = 0,

bkgr_var = 1,

hub_size = [2,4,6,8,10]

hub_mean = [0.5,1,1.5,2,2.5,3,3.5,4]

hub_var = 1

seed = [0,1,2,3,4,5,6,7,8,9,10] #,6,7,8,9,10,11,12,13,14]

seed_pairs = [f'{i[0]}-{i[1]}' for i in itertools.combinations(seed,2)]
