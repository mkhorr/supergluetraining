import numba as nb
import numpy as np


#https://github.com/numba/numba/issues/1269
@nb.njit
def np_apply_along_axis(func1d, axis, arr, dtype=np.int64):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def ground_truth_matches(kpts0, kpts1, K0, K1, T0to1, depth0, depth1):
    
    m = kpts0.shape[0]
    n = kpts1.shape[0]

    # compute reprojection matrix
    depths0 = np.full(fill_value=np.nan,shape=(m,1), dtype=np.float32)
    for i in range(m):
        depths0[i][0] = depth0[np.int32(kpts0[i][1]), np.int32(kpts0[i][0])]
    kpts0to1 = (K1 @ T0to1 @ np.linalg.inv(K0)) @ np.concatenate((kpts0 * depths0, depths0, np.ones((m,1), dtype=np.float32)), axis=1).T
    depths0to1 = np.expand_dims(kpts0to1.T[:, 2], 1)
    kpts0to1 = (kpts0to1.T / depths0to1)[:,:2]

    depths1 = np.full(fill_value=np.nan,shape=(n,1), dtype=np.float32)
    for i in range(n):
        depths1[i][0] = depth1[np.int32(kpts1[i][1]), np.int32(kpts1[i][0])]
    kpts1to0 = (K0 @ np.linalg.inv(T0to1) @ np.linalg.inv(K1)) @ np.concatenate((kpts1 * depths1, depths1, np.ones((n,1), dtype=np.float32)), axis=1).T
    depths1to0 = np.expand_dims(kpts1to0.T[:, 2], 1)
    kpts1to0 = (kpts1to0.T / depths1to0)[:,:2]

    match_indices0 = np.full((401,1), -1, dtype=np.int64)
    match_indices1 = np.full((401,1), -1, dtype=np.int64)
    match_weights0 = np.full((401,1), 0, dtype=np.float32)
    match_weights1 = np.full((401,1), 0, dtype=np.float32)

    errs = np.sqrt(np.sum((np.expand_dims(kpts1to0,1) - kpts0)**2, axis=2)).T
    errs += np.sqrt(np.sum((np.expand_dims(kpts0to1,1) - kpts1)**2, axis=2))
    errs /= 2.0

    i_mins = np_apply_along_axis(np.argmin, 1, errs)
    j_mins = np_apply_along_axis(np.argmin, 0, errs)

    # checking for matches
    match_cnt = 0
    drop_cnt = 0
    for i in range(m):
        j = i_mins[i]
        if j_mins[j] == i and errs[i, j] <= 10 and depths1[j][0] != 0 and abs(depths0to1[i][0] - depths1[j][0]) / depths1[j][0] <= .1:
            match_indices0[i] = j
            match_indices1[j] = i
            match_cnt += 1
        elif depths0[i][0] < 1e-6 or errs[i][j] <= 15:
            match_weights0[i] = -1 #flag - will be set to 0
            drop_cnt += 1
 
    for j in range(n):
        i = j_mins[j]
        if match_indices1[i] != -1 and (depths1[j][0] < 1e-6 or errs[i][j] <= 15):
            match_weights1[j] = -1 #flag - will be set to 0
            drop_cnt += 1

    if match_cnt == 0 or match_cnt == (m + n - drop_cnt)/2:
        return match_indices0, match_indices1, match_weights0, match_weights1

    # weigh entries to balance classes
    matched_weight = 2 * match_cnt / (m + n - drop_cnt)
    unmatched_weight = .5 / (1 - matched_weight)
    matched_weight = .5 / matched_weight
    for i in range(m):
        if match_weights0[i] == -1:
            match_weights0[i] = 0
        else:
            match_weights0[i] = unmatched_weight if match_indices0[i] == -1 else matched_weight
    for j in range(n):
        if match_weights1[j] == -1:
            match_weights1[j] = 0
        else:
            match_weights1[j] = unmatched_weight if match_indices1[j] == -1 else matched_weight
    return match_indices0, match_indices1, match_weights0, match_weights1
