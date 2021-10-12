import numpy as np
import pandas as pd



def worker(order):
    # to compute some statistics
    rows, trace = order
    ans = np.zeros((len(rows), 5))
    for i, r in enumerate(rows):
        mtrace = trace[r[0]: r[1]]
        ans[i, :] = [mtrace.mean(), np.median(mtrace), np.std(mtrace), mtrace.min(), mtrace.max()]
    return ans


def regionprops1d(mask, *_trace, n_proc=1):
    """
    :param mask: 1d numpy array mask of bool, or [0,1]
    :param _trace: traces to process for stats (max, min, std, median, mean)
    :param n_proc: number of processes to spawn while calculating stats for traces
    :return: DataFrame with indices Onset (inclusive), Offset (exclusive), Length (absolute)
             When given _trace, also calculated Min, Max, STD, Median, and Mean for each trace
    """

    assert (len(mask.shape) == 1), 'shape mismatch, mask should be 1d'

    col_names = ['Onset', 'Offset', 'Length']

    if not (mask.any()):
        return None

    diff_mask = np.diff(mask, axis=-1, prepend=0, append=0)
    t_start = np.where(diff_mask == 1)[0]
    t_stop = np.where(diff_mask == -1)[0]
    assert (len(t_start) == len(t_stop))

    res = np.vstack([t_start, t_stop, t_stop - t_start]).T
    df_res = pd.DataFrame(res, columns=col_names, dtype=int)

    for itr, trace in enumerate(_trace):
        assert(len(trace) == len(mask)), f'trace len = {len(trace)} not equal to mask len = {len(mask)}'

        order = [(s, trace) for s in np.array_split(res, n_proc, axis=0)]

        if itr:
            cols = list(map(lambda x: f'{x}_{itr}', ['Mean', 'Median', 'STD', 'Min', 'Max']))
        else:
            cols = ['Mean', 'Median', 'STD', 'Min', 'Max']

        if n_proc > 1:
            with Pool(n_proc) as p:
                ans = pd.DataFrame(np.vstack(p.map(worker, order)), columns=cols)
        else:
            ans = pd.DataFrame(np.vstack(list(map(worker, order))), columns=cols)

        df_res = pd.merge(left=df_res, right=ans, left_index=True, right_index=True)

    return df_res

