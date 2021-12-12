import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from proteingnn.data import pssm2D
from proteingnn.example.util import visual_pssm_corr

plt.style.use('ggplot')
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"])


def main():
    benchmark_dir = Path('unsupervised')
    if not benchmark_dir.exists():
        benchmark_dir.mkdir()

    # collect pssm benchmark
    csv_dir = Path('data/csv')
    pssm_src_dir = Path('data/pssm')

    csv_dict = {csv.stem: csv for csv in csv_dir.glob('*.csv')}

    # collect pssm performance
    df_pssm = []

    for pssm in pssm_src_dir.glob('*.pssm'):
        dataset_name = pssm.stem
        pdb_code = dataset_name.replace('_', '')

        print(f'\rProcess {dataset_name}.pssm' + ' '*50, end='')

        if dataset_name not in csv_dict:
            print(f'{dataset_name}.csv not found.')
            continue

        csv = csv_dict[dataset_name]
        if not csv.exists():
            print(f'{csv} not found.')
            continue

        # exp_data as dictionary
        df = pd.read_csv(csv)
        df = df.set_index('mutant')
        df = df.loc[[i for i in df.index if i not in ('WT', 'wt')]]
        exp_data = df['exp'].dropna().to_dict()

        benchmark_subdir = benchmark_dir / pssm.stem
        if not benchmark_subdir.exists():
            benchmark_subdir.mkdir()

        visual_pssm_corr(
            pssm=pssm_src_dir / f'{dataset_name}.pssm',
            exp_data=exp_data,
            rootdir=benchmark_subdir,
            alpha=0.1
        )
        dpssm = pssm2D(pssm=pssm, return_type='OrderedDict', relative=True)
        dpssm = pd.DataFrame({'pssm': dict(dpssm.items())})
        df = df.join(pd.DataFrame(dpssm))

        df = df.rename(columns={
            'mutation_effect_prediction_vae_ensemble': 'DeepSequence',
            'mutation_effect_prediction_pairwise': 'EVMutation',
            'pssm': 'PSSM'
        })

        cols = ['exp', 'DeepSequence', 'EVMutation', 'PSSM']
        df_pcc = df[cols].corr('pearson')['exp']
        df_src = df[cols].corr('spearman')['exp']
        df_corr = pd.DataFrame([df_pcc, df_src])
        del df_corr['exp']

        df_corr = df_corr.T
        df_corr.columns = ['pcc', 'src']
        df_corr['dataset'] = dataset_name

        df_pssm.append(df_corr)

    df_pssm = pd.concat(df_pssm)
    df_pssm.to_csv(benchmark_dir / 'pssm.csv')

    # visualization of all unsupervised methods
    metric = 'src'
    embedding_radius = 'esm-6'

    df = []
    for csv in benchmark_dir.glob('*/ESM.csv'):
        df_dataset = pd.read_csv(csv)
        df_dataset['dataset'] = csv.parent.name
        df.append(df_dataset)

    df = pd.concat(df)
    df = df[df['metric'] == 'masked-marginals']
    df = df.groupby('dataset').mean()
    df = df.reset_index()
    df['model'] = 'ESM-1v'

    df2 = df_pssm.reset_index()
    df2 = df2.rename(columns={'index': 'model'})
    df2 = df2.append(df[['dataset', 'model', 'src', 'pcc']])

    csv = Path(f'{embedding_radius}/test_{metric}.csv')
    if not csv.exists():
        raise FileNotFoundError(f'{csv} not found.')

    df3 = pd.read_csv(csv)
    df3 = df3[['dataset', 'SingleSiteMLP']]
    df3 = df3.rename(columns={'SingleSiteMLP': metric})
    df3['model'] = 'SingleSiteMLP'
    df3 = df3.append(df2).append(df)

    df4 = pd.pivot_table(df3, values='src', index='dataset', columns='model')
    df4 = df4.abs()
    idx = np.argsort(df4['SingleSiteMLP'].values)
    df4 = df4.iloc[idx]

    models_ordered = ['SingleSiteMLP', 'DeepSequence', 'PSSM', 'EVMutation', 'ESM-1v']
    df4 = df4[models_ordered]
    df4 = df4.dropna()
    df5 = df4.reset_index()
    df5.to_csv(benchmark_dir / 'unsupervised.csv')

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, model in enumerate(models_ordered, 1):
        shift = 0.1 * (i - (len(models_ordered) + 1) / 2)
        ax.scatter(x=np.arange(len(df5)) + shift, y=df5[model], s=20, label=model)

    ax.set_ylim(0, None)
    ax.set_xticks(np.arange(len(df5)))
    ax.set_xticklabels(df5['dataset'], rotation=45, fontsize=8, ha='right')
    for xticklabel in ax.get_xticklabels():
        if xticklabel.get_text() in ('BG_STRSQ',):
            xticklabel.set_fontweight('bold')

    ax.set_xlabel('')
    ax.set_ylabel('Spearman Rank Correlation', fontsize=10)
    ax.legend()
    plt.tight_layout()

    fig = ax.get_figure()
    fig.savefig(benchmark_dir / 'unsupervised.png', dpi=300)


if __name__ == '__main__':
    main()
