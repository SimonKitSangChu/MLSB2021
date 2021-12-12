from glob import glob
import pandas as pd
from pathlib import Path
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch

from proteingnn.util import pdb2distance_map

plt.style.use('ggplot')


def fill_df_bins(df, max_bin, bin_name='distance_bin'):
    for bin_ in range(1, max_bin+1):
        if bin_ not in df[bin_name].to_list():
            df = df.append({'distance_bin': bin_, 'p': 0}, ignore_index=True)
            
    return df.sort_values(bin_name, ascending=True)


def main():
    dataset_rootdir = Path('datasets')
    embedding_rootdir = Path('data/embeddings')
    structure_rootdir = Path('data/alphafold2')

    embedding_dirs = list(embedding_rootdir.glob('*/esm'))
    for dataset_i, embedding_dir in enumerate(embedding_dirs, 1):
        dataset_name = embedding_dir.parent.name

        csv = Path(f'esm_locality_{dataset_name}.csv')
        if not csv.exists():
            pdb = structure_rootdir / dataset_name / 'ranked_0.pdb'
            if not pdb.exists():
                continue

            dist_mat = pdb2distance_map(pdb)
            df = []

            pts = [pt for pt in embedding_dir.glob('*.pt')
                   if 'indices' not in pt.stem and 'wildtype' not in pt.stem]
            for i, pt in enumerate(pts, 1):
                if dataset_name == 'PABP_doubles':
                    continue

                print(f'\rProgress {dataset_name} [{dataset_i}|{len(embedding_dirs)}] [{i}|{len(pts)}]', end='')

                mut_name = pt.stem.split('_')[-1]
                resid_mt = int(mut_name[1:-1])
                layer = torch.load(pt)
                norm = layer.norm(dim=1) # norm of each mutational embedding
                pdf = norm / norm.norm(dim=0)  # normalization across nodes

                # binning in sequence and distance separation
                for j, p in enumerate(pdf):
                    resid = j + 1
                    seq_sep = abs(resid_mt - resid)
                    dist = dist_mat[resid-1][resid_mt-1].item()
                    df.append({
                        'mut_name': mut_name,
                        'resid': resid,
                        'sequence separation': seq_sep,
                        'distance': dist,
                        'p': p.item(),
                        'dataset_name': dataset_name
                    })

                del layer, norm, pdf

            df = pd.DataFrame(df)
            df.to_csv(csv, index=False)

    # visualization
    min_seq_sep = 15
    max_dis_sep = 25

    distance_bins = np.linspace(0, 24, 24)
    sequence_bins = np.arange(0, 20, 1)

    for csv in glob('esm_locality_*.csv'):
        df = pd.read_csv(csv, index_col=False)
        dataset_name = csv.replace('.csv', '').replace('esm_locality_', '')

        df['distance_bin'] = np.digitize(df['distance'], distance_bins)
        df['sequence_bin'] = np.digitize(df['sequence separation'], sequence_bins)

        # mean statistics
        df_dis = df.groupby(['distance_bin'])['p'].mean()
        df_seq = df.groupby(['sequence_bin'])['p'].mean()

        df_dis_cut = df[(df['sequence separation'] > min_seq_sep) & (df['distance'] < max_dis_sep)]
        df_dis_cut = df_dis_cut.groupby(['distance_bin'])['p'].mean()

        # visualization
        # df = df_seq.reset_index()
        # df.plot.bar(x='sequence_bin', y='p', legend=None)

        df = df_dis.reset_index()
        df = fill_df_bins(df, max_bin=24, bin_name='distance_bin')
        ax = df.plot.bar(x='distance_bin', y='p', legend=None, color='tab:blue')

        df = df_dis_cut.reset_index()
        df = fill_df_bins(df, max_bin=24, bin_name='distance_bin')
        ax = df.plot.bar(x='distance_bin', y='p', legend=None, ax=ax, xlabel=r'Distance ($\AA$)', ylabel='Normalized embedding norm',
                         color='tab:red')
        ax.get_figure().savefig(f'mut_norm_{dataset_name}.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    main()

