from absl import app, flags
from Bio import SeqIO
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from proteingnn.data import *
from proteingnn.util import plot_corr

FLAGS = flags.FLAGS
flags.DEFINE_string('esm_install_path', None, 'Github location of ESM install.')
flags.DEFINE_boolean('verbose', False, 'Verbose mode.')
flags.DEFINE_string('stage', 'postprocess', 'Options: preprocess/postprocess')
flags.mark_flag_as_required('stage')


def main(argv):
    if FLAGS.stage == 'preprocess' and FLAGS.esm_install_path is None:
        raise FileNotFoundError('Requires path to ESM github for ESM-1v benchmark.')

    benchmark_dir = Path('unsupervised')
    if not benchmark_dir.exists():
        benchmark_dir.mkdir()

    csv_dir = Path('data/csv')
    csv_dict = {csv.stem: csv for csv in csv_dir.glob('*.csv')}

    cmds = []

    fasta_dir = Path('data/fasta')
    for fasta in fasta_dir.glob('*.fasta'):
        dataset_name = fasta.stem
        pdb_code = dataset_name.replace('_', '')

        print(f'\rProceed to {dataset_name}', end='')

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

        benchmark_subdir = benchmark_dir / fasta.stem
        if not benchmark_subdir.exists():
            benchmark_subdir.mkdir()

        if FLAGS.stage == 'preprocess':
            # exp_data as dataframe
            csv_input = benchmark_subdir / 'input.csv'
            df.to_csv(csv_input)

            # run esm predictions; notes:
            #  skipped pseudo-ppl
            #  offset_idx 1 means no offset
            sequence = SeqIO.read(fasta, 'fasta').seq

            cmd = f'''for scoring in masked-marginals wt-marginals; do
    if [ ! -f {benchmark_subdir}/ESM-1v_$scoring.csv ]; then
        python {FLAGS.esm_install_path}/variant-prediction/predict.py \\
            --model-location esm1v_t33_650M_UR90S_1 esm1v_t33_650M_UR90S_2 esm1v_t33_650M_UR90S_3 esm1v_t33_650M_UR90S_4 esm1v_t33_650M_UR90S_5 \\
            --sequence {sequence} \\
            --dms-input {csv_input} \\
            --mutation-col mutant \\
            --dms-output {benchmark_subdir}/ESM-1v_$scoring.csv \\
            --offset-idx 1 \\
            --scoring-strategy $scoring
    fi
done
'''
            cmds.append(cmd)

        elif FLAGS.stage == 'postprocess':
            models = [
                'esm1v_t33_650M_UR90S_1',
                'esm1v_t33_650M_UR90S_2',
                'esm1v_t33_650M_UR90S_3',
                'esm1v_t33_650M_UR90S_4',
                'esm1v_t33_650M_UR90S_5',
            ]

            # visualization
            df_subset = []
            for metric in ('masked-marginals', 'wt-marginals'):
                csv = benchmark_subdir / f'ESM-1v_{metric}.csv'
                try:
                    df = pd.read_csv(csv).set_index('mutant')
                    df = df.dropna(subset=['exp'] + models)
                except FileNotFoundError:
                    warnings.warn(f'{csv} not found. Skip ESM-1v benchmark.')

                try:
                    df[models]
                except KeyError:
                    warnings.warn(f'esm1v prediction not found in {csv}. Potential issues are wrong mutations or '
                                  f'sequence length exceeds limit.')
                    continue

                for model in models:
                    plot_corr(
                        x=df[model],
                        y=df['exp'],
                        x_name=model,
                        y_name='exp',
                        png=f'exp-{model}.png',
                        rootdir=benchmark_subdir,
                        alpha=0.1
                    )

                    datum = (
                        metric,
                        model,
                        pearsonr(df['exp'], df[model])[0],
                        spearmanr(df['exp'], df[model])[0],
                    )
                    df_subset.append(datum)

            df_subset = pd.DataFrame(df_subset, columns=['metric', 'model', 'pcc', 'src'])
            df_subset.to_csv(benchmark_subdir / 'ESM.csv')

        else:
            raise ValueError(f'Option {FLAGS.stage} not available.')

    if FLAGS.stage == 'preprocess':
        with open('benchmark_esm.sh', 'w') as f:
            f.write('\n'.join(cmds))
            f.write('echo "Please finish benchmark with postprocess option in ESM.py"\n')

        print('Please run benchmask_esm.sh to benchmark ESM-1v.')


if __name__ == '__main__':
    app.run(main)
