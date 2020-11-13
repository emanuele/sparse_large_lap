import numpy as np
from matching import min_weight_full_bipartite_matching
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from time import time


def simulate_sparse(n_rows, n_columns, row_density=0.001):
    """For each row, generate non-zero values only on certain columns,
    according to the given row_density.
    """
    # X = sparse.lil_matrix((n_rows, n_columns))
    X = []
    counter = 0
    for row in range(n_rows):
        counter += 1
        if (counter % 1000) == 0:
            print(counter)

        # X[row] = sparse.random(1, n_columns, density=row_density)
        X.append(sparse.random(1, n_columns, density=row_density, format='lil'))

    return sparse.vstack(X).tocsr()


if __name__=='__main__':

    n_rows = 950_000
    n_columns = 950_000
    density = 4.43e-5

    lsa = False

    print('Creating the cost matrix: %s x %s (density=%s)' % (n_rows, n_columns, density))
    cost = sparse.random(n_rows, n_columns, density=density, format='csr')
    # cost = simulate_sparse(n_rows, n_columns, row_density=density)
    print('%s nonzero entries' % cost.getnnz())

    if lsa:
        cost_dense = cost.todense()
        row_ind, col_ind = linear_sum_assignment(cost_dense)
        total_cost = cost_dense[row_ind, col_ind].sum()
        print('lsa cost=%s' % total_cost)

    print('MWFBM')
    t0 = time()
    row_ind2, col_ind2 = min_weight_full_bipartite_matching(cost)
    print('%s sec' % (time() - t0))
    total_cost2 = cost[row_ind2, col_ind2].sum()
    print('mwfb cost=%s' % total_cost2)
