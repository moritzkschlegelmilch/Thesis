import pandas as pd

def print_tuple_dict_matrices(push, pull, decimals=4):
    """
    Print two tuple-key dictionaries as matrices in the terminal.

    Each input must look like:
        {('row_name', 'col_name'): value, ...}
    """
    def to_matrix(d):
        if not isinstance(d, dict):
            raise TypeError("Each input must be a dictionary.")
        if not all(isinstance(k, tuple) and len(k) == 2 for k in d):
            raise ValueError("All keys must be 2-item tuples like ('row', 'col').")

        df = pd.Series(d).unstack()

        # Keep a stable row/column order based on appearance in the dict
        rows = []
        cols = []
        for r, c in d.keys():
            if r not in rows:
                rows.append(r)
            if c not in cols:
                cols.append(c)

        return df.reindex(index=rows, columns=cols)

    m1 = to_matrix(push).round(decimals)
    m2 = to_matrix(pull).round(decimals)

    print("\nPush:")
    print(m1.to_string())

    print("\nPull:")
    print(m2.to_string())

    return m1, m2