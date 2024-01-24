import pandas as pd


def samples_to_sample_max(df):
    print(df)
    # for t in t_vals:
    #     for K in range(K_max):
    #         tempdf = df[(df['t'] == t) & (df['K'] == K)]
    #         print(tempdf)
    #         max_val = tempdf[]
    out = df.groupby(['sched', 'K']).max()
    # print(df.groupby(['t', 'K']).max())
    out.reset_index().to_csv('data/sample_max.csv', index=False)


def main():
    # sdp_df = pd.read_csv('data/NNLS_spreadt_halfc.csv')
    samples_df = pd.read_csv('data/sample_data.csv')

    samples_to_sample_max(samples_df)


if __name__ == '__main__':
    main()
