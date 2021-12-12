from absl import app
from absl import flags
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import warnings

plt.style.use('ggplot')
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"])

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('dataset', None, 'Dataset(s) to assess. (Default: all datasets)')
flags.DEFINE_string('rootdir', 'esm-6', 'Root directory for supervisied model benchmark.')
flags.DEFINE_multi_string('highlight_dataset', 'BG_STRSQ', 'Dataset(s) to highlight in visualization.')


def main(argv):
    rootdir = Path(FLAGS.rootdir)

    if FLAGS.dataset:
        dataset_names = FLAGS.dataset.copy()
    else:
        csv_dir = Path('data/csv')
        csv_dict = {csv.stem: csv for csv in csv_dir.glob('*.csv')}
        dataset_names = sorted(csv_dict.keys())

    # visualization of all datasets
    mean_dict = {'train': [], 'val': [], 'test': []}
    sem_dict = {'train': [], 'val': [], 'test': []}
    pvalue_dict = {'vanilla': {}, 'Bonferroni': {}, 'record': []}

    for dataset_name in dataset_names:
        if dataset_name == 'PABP_doubles':  # drop double mutant
            continue

        print(f'\r{dataset_name}' + ' '*50, end='')

        benchmark_dir = rootdir / dataset_name
        if not benchmark_dir.exists():
            if FLAGS.dataset:
                raise FileNotFoundError(f'Prespecificed {dataset_name} not exists.')
            continue

        cols_eval = ['pcc', 'src', 'r2', 'subdataset']
        cols_drop = ['betas', 'alphal2', 'norm_name', 'gnn_class', 'gnorm_class', 'Unnamed: 0',
                     'pcc_p', 'src_p', 'mse']

        baseline_names = {
            'eval_DummyGraphConv': 'NoEdgeGCN',
            'eval_FastMLP': 'WholeSeqMLP',
            'eval_SeqPoolingMLP': 'SeqPoolingMLP',
            'eval_SingleSiteMLP': 'SingleSiteMLP',
            'eval_GCNConv': 'GCNConv',
            'eval_SAGEConv': 'SAGEConv',
            'eval_GATConv': 'GATConv',
        }

        # aggregate best model from each baseline model types
        dfs = []
        for csv in benchmark_dir.glob('eval*.csv'):
            if csv.stem not in baseline_names:
                continue

            if '_best.csv' in csv.name:
                continue

            baseline_name = baseline_names[csv.stem]
            df = pd.read_csv(csv)
            if 'Unnamed: 0' in df.columns:
                del df['Unnamed: 0']
            df_val = df[df['subdataset'] == 'val']

            cols_hparam = [col for col in df_val.columns if col not in cols_eval and col not in cols_drop]
            df_model = df_val.groupby(cols_hparam).mean()
            df_model = df_model['r2'].nlargest(1).reset_index()

            df_tops = df[(df[cols_hparam] == df_model[cols_hparam].iloc[0]).all(axis=1)]
            df_tops.to_csv(str(csv).replace('.csv', '_best.csv'), index=False)

            df_tops = df_tops[cols_eval].copy()
            df_tops['model_type'] = baseline_name

            dfs.append(df_tops)

        dfs = pd.concat(dfs)
        dfs.to_csv(benchmark_dir / 'performance.csv', index=False)

        # summary statistics and visualization
        for subdataset, df_sub in dfs.groupby('subdataset'):
            # sort by pcc
            df_mean = df_sub.groupby('model_type').mean()
            df_sem = df_sub.groupby('model_type').sem()

            model_types = df_mean['pcc'].sort_values(ascending=False).index
            df_mean = df_mean.loc[model_types]
            df_sem = df_sem.loc[model_types]

            ax = df_mean.plot(kind='bar', yerr=df_sem, rot=0, xlabel='')

            plt.tight_layout()
            fig = ax.get_figure()
            fig.savefig(benchmark_dir / f'compare_{subdataset}.png', dpi=300)
            plt.close()

            # save to dataset-wise visualization
            df_mean['dataset'] = dataset_name
            mean_dict[subdataset].append(df_mean)
            df_sem['dataset'] = dataset_name
            sem_dict[subdataset].append(df_sem)

        # statistical testing
        subdataset = 'test'
        target_type = 'GCNConv'
        metrics_sides = {
            'pcc': 'greater',
            'src': 'greater',
            'r2': 'greater'
        }

        df = dfs[dfs['subdataset'] == subdataset]
        df_target = df[df['model_type'] == target_type]

        df_pvalue = {}
        for model_type, df_model in df.groupby('model_type'):
            if model_type != target_type:
                metric_tests = {}

                for metric, side in metrics_sides.items():
                    statistic, pvalue = wilcoxon(x=df_target[metric], y=df_model[metric], alternative=side)
                    metric_tests[metric] = pvalue

                df_pvalue[model_type] = metric_tests

        df_pvalue = pd.DataFrame(df_pvalue)
        df_pvalue.to_csv(benchmark_dir / 'performance_pvalues.csv')

        pvalue_dict['vanilla'][dataset_name] = (df_pvalue < 0.05).all(axis=1)
        pvalue_dict['Bonferroni'][dataset_name] = (df_pvalue < (0.05 / df_pvalue.count())).all(axis=1)

        df_pvalue = df_pvalue.melt(var_name='model', value_name='pvalue')
        df_pvalue['dataset'] = dataset_name
        pvalue_dict['record'].append(df_pvalue)

    # save pvalue analysis of all datasets
    #  is there any scenario where GCNConv > all other baselines?
    df_vanilla = pd.DataFrame(pvalue_dict['vanilla'])
    df_bon = pd.DataFrame(pvalue_dict['Bonferroni'])
    df_record = pd.concat(pvalue_dict['record'])
    df_vanilla.to_csv(rootdir / 'pvalue_vanilla.csv')
    df_bon.to_csv(rootdir / 'pvalue_Bonferroni.csv')
    df_record.to_csv(rootdir / 'pvalue_record.csv')

    metric_name_dict = {
        'pcc': 'Pearson Correlation Coefficient',
        'src': 'Spearman Rank Correlation',
        'r2': 'R-square'
    }

    # visualization of all datasets
    for subdataset in mean_dict.keys():
        for metric_name in ('pcc', 'src', 'r2'):
            df_mean = mean_dict[subdataset].copy()
            df_sem = sem_dict[subdataset].copy()

            if len(df_mean) == 0:
                warnings.warn(f'Empty metric evaluation dataframe {subdataset} {metric_name}.')
                continue
            elif len(df_mean) == 1:
                df_mean = df_mean[0]
                df_sem = df_sem[0]
            else:
                df_mean = pd.concat(df_mean)
                df_sem = pd.concat(df_sem)

            df_mean = df_mean.reset_index()
            df_mean = pd.pivot_table(df_mean, values=metric_name, index='dataset', columns='model_type')
            df_sem = df_sem.reset_index()
            df_sem = pd.pivot_table(df_sem, values=metric_name, index='dataset', columns='model_type')

            if 'GCNConv' in df_mean.columns:
                models_ordered = ['SingleSiteMLP', 'GCNConv', 'NoEdgeGCN', 'SeqPoolingMLP']
            else:
                models_ordered = ['SingleSiteMLP', 'GCNModel', 'NoEdgeGCN', 'SeqPoolingMLP']
            df_mean = df_mean[models_ordered]

            if df_sem.empty:
                df_sem = df_mean.copy()
                df_sem.loc[:, :] = float('nan')
            else:
                df_sem = df_sem[models_ordered]

            rename_dict = {'GCNConv': 'GCNModel'}
            df_mean = df_mean.rename(columns=rename_dict)
            df_sem = df_sem.rename(columns=rename_dict)

            idx = np.argsort(df_mean['SingleSiteMLP'].values)
            df_mean = df_mean.iloc[idx].reset_index()
            df_sem = df_sem.iloc[idx].reset_index()

            # figsize = (3, 5)
            figsize = (9, 5)
            fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=figsize, gridspec_kw={'height_ratios': [1, 4]})
            for i, model in enumerate(df_mean.columns, 1):
                if model == 'dataset':
                    continue

                shift = 0.1 * (i - (len(models_ordered) + 1) / 2)
                ax0.errorbar(x=np.arange(len(df_mean))+shift, y=df_mean[model]-df_mean['GCNModel'],
                             yerr=df_sem[model], fmt='.', markersize=9, label=model, alpha=0.8)
                ax1.errorbar(x=np.arange(len(df_mean))+shift, y=df_mean[model],
                             yerr=df_sem[model], fmt='.', markersize=9, label=model, alpha=0.8)

            ax0.set_xticks(np.arange(len(df_mean)))
            ax0.tick_params(axis='x', length=0)
            ax0.set_xticklabels([])
            ax0.set_ylim(-0.25, 0.25)
            ax0.set_yticks([0])
            ax0.set_yticklabels([0])

            ax1.set_ylim(0, None)
            ax1.set_xticks(np.arange(len(df_mean)))
            ax1.set_xticklabels(df_mean['dataset'], rotation=45, fontsize=8, ha='right')
            for xticklabel in ax1.get_xticklabels():
                if xticklabel.get_text() in FLAGS.highlight_dataset:
                    xticklabel.set_fontweight('bold')

            ax1.set_ylabel(metric_name_dict[metric_name], fontsize=9)
            ax1.legend()
            plt.tight_layout()

            fig.savefig(f'{FLAGS.rootdir}/{subdataset}_{metric_name}.png', dpi=300)
            plt.close()

            df = df_mean.join(df_sem.drop(columns='dataset'), rsuffix='_sem')
            df.to_csv(f'{FLAGS.rootdir}/{subdataset}_{metric_name}.csv')


if __name__ == '__main__':
    app.run(main)
