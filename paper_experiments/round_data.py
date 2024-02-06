import numpy as np
import pandas as pd


def round_dfs(in_files, out_files):
    skip_cols = [0, 1]
    middle_cols = [2, 3, 4, 5, 6, 7]
    # middle_cols = [2, 3, 4, 5]
    # percent_cols = [6, 7]
    percent_cols = []

    def sigfigs_to_latex(input_str):
        print(input_str)
        num = input_str.split('e')
        print(num)
        return f'${num[0]} \\times 10^{{{int(num[1])}}}$'

    def process(val):
        if np.isnan(val):
            return ''
        if 1 <= val and val < 10:
            # return str(np.round(val, 2))
            return f'{np.round(val, 2):.2f}'
        else:
            return sigfigs_to_latex(f'{val:.2e}')

    def process_vec(vals):
        return np.vectorize(process)(vals)

    def process_percents(val):
        if np.isnan(val):
            return ''
        thresh = 99.99
        if val <= 99.99:
            # return str(np.round(val, 2))
            return f'{np.round(val, 2):.2f}'
        else:
            return f'${thresh}^{{(\star)}}$'

    def process_percent_vec(vals):
        return np.vectorize(process_percents)(vals)

    # out_df = []
    for in_f, out_f in zip(in_files, out_files):
        if 'glob' in in_f:
            skip_cols = [0, 1]
            middle_cols = [2, 3, 4, 5]
        elif 'silver' in in_f:
            skip_cols = [0, 2]
            middle_cols = [1, 3, 4, 5, 6, 7, 8]
        else:
            skip_cols = [0, 1]
            middle_cols = [2, 3, 4, 5, 6, 7]

        new_df = pd.DataFrame()
        df = pd.read_csv(in_f)
        cols = df.columns
        print(cols)
        len(df.columns)

        for col_idx in skip_cols:
            col = cols[col_idx]
            new_df[col] = df[col]
        for col_idx in middle_cols:
            col = cols[col_idx]
            single_col = df[col]
            print(process_vec(single_col))
            new_df[col] = process_vec(single_col)
        for col_idx in percent_cols:
            col = cols[col_idx]
            single_col = df[col]
            print(process_percent_vec(single_col))
            new_df[col] = process_percent_vec(single_col)
        new_df.to_csv(out_f, index=False)
        print(new_df)


def main():
    # in_files = ['NNLS/data/NNLS_papertab_raw.csv']
    # out_files = ['NNLS/data/roundtest.csv']
    # in_files = [
    #     'NNLS/data/NNLS_papertab_raw.csv',
    #     'silver/data/silver_papertab_raw.csv',
    #     'ISTA/data/ISTA_papertab_raw.csv',
    #     'NUM/data/NUM_papertab_raw.csv',
    #     'MPC/data/MPC_papertab_raw.csv',
    #     'ISTA/data/ISTA_glob_papertab.csv',
    # ]
    # out_files = [
    #     'NNLS/data/NNLS_roundsci.csv',
    #     'silver/data/silver_roundsci.csv',
    #     'ISTA/data/Lasso_roundsci.csv',
    #     'NUM/data/NUM_roundsci.csv',
    #     'MPC/data/MPC_roundsci.csv',
    #     'ISTA/data/ISTA_glob_roundsci.csv',
    # ]
    in_files = [
        'NNLS/data/NNLS_paper_ratio.csv',
        'NNLS/data/NNLS_nonstrong_raw_pt1.csv',
        'NNLS/data/NNLS_nonstrong_raw_pt2.csv',
        'silver/data/strongsilver_paper_ratio.csv',
        'silver/data/nonstrongsilver_paper_ratio.csv',
        'NUM/data/NUM_paper_ratio.csv',
        'ISTA/data/ISTA_paper_ratio.csv',
        'MPC/data/MPC_paper_ratio.csv',
    ]

    out_files = [
        'NNLS/data/NNLS_ratio_rounded.csv',
        'NNLS/data/NNLS_nonstrong_rounded_pt1.csv',
        'NNLS/data/NNLS_nonstrong_rounded_pt2.csv',
        'silver/data/strongsilver_ratio_rounded.csv',
        'silver/data/nonstrongsilver_ratio_rounded.csv',
        'NUM/data/NUM_ratio_rounded.csv',
        'ISTA/data/ISTA_ratio_rounded.csv',
        'MPC/data/MPC_ratio_rounded.csv'
    ]
    round_dfs(in_files, out_files)


if __name__ == '__main__':
    main()
