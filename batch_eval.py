from absl import app
from absl import flags
from collections import defaultdict
import pandas as pd
import warnings

from proteingnn.data import BaseDatamodule
from proteingnn.example.data import DefaultDatamodule, SingleSiteDataset
from proteingnn.example.model import *
from proteingnn.util import get_checkpoints, evaluate_ckpt

FLAGS = flags.FLAGS
flags.DEFINE_string('rootdir', None, 'Root directory to locate ckpt.')
flags.DEFINE_boolean('verbose', False, 'Verbose mode.')
flags.DEFINE_string('dataset', None, 'Dataset used in evaluation.')
flags.DEFINE_multi_string('subdataset', 'val', 'Sub-dataset evaluated. Options are train/val/test.')
flags.DEFINE_string('model_class', None, 'Options: FastGCNModel, SingleSiteMLP, FastMLPModel, SeqPoolingMLP')
flags.DEFINE_string('embedding_radius', 'esm-6', '{Embedding name}-{radius}')
flags.mark_flags_as_required(['model_class', 'dataset'])


def main(argv):
    dataset_name = FLAGS.dataset
    pdb_code = dataset_name.replace('_', '')
    embedding_name, radius = FLAGS.embedding_radius.split('-')

    classes = defaultdict(lambda: FastGCNModel)
    classes['SingleSiteMLP'] = SingleSiteMLP
    classes['DummyGraphConv'] = FastGCNModel
    classes['GCNConv'] = FastGCNModel
    classes['FastMLP'] = FastMLPModel
    classes['SeqPoolingMLP'] = SeqPoolingMLP

    if FLAGS.model_class not in classes:
        warnings.warn(f'{FLAGS.model_class} not recongized. Use default {FastGCNModel}.')
    class_ = classes[FLAGS.model_class]

    # create dataset
    csv_dir = Path('data/csv')
    csv_dict = {csv.stem: csv for csv in csv_dir.glob('*.csv')}

    df = pd.read_csv(csv_dict[dataset_name])
    df = df.set_index('mutant')
    df = df.loc[[i for i in df.index if i not in ('WT', 'wt')]]
    exp_data = df['exp'].dropna().to_dict()

    if FLAGS.model_class == 'SingleSiteMLP':
        dataset = SingleSiteDataset(
            root=f'data/embeddings/{dataset_name}/{embedding_name}',
            exp_data=exp_data, pdb_code=pdb_code
        )
        datamodule = BaseDatamodule(
            dataset=dataset, batch_size=1, pin_memory=False, num_workers=1,
            split_root=f'datasets/{dataset_name}'
        )
    else:
        datamodule = DefaultDatamodule(
            root=f'datasets/{dataset_name}/{embedding_name}-{radius}', exp_data=exp_data,
            batch_size=1, pin_memory=False, num_workers=1, split_root=f'datasets/{dataset_name}'
        )

    df_logging = []
    for subdataset_name in FLAGS.subdataset:
        if subdataset_name == 'train':
            subdataset = datamodule.train_set
        elif subdataset_name == 'val':
            subdataset = datamodule.val_set
        elif subdataset_name == 'test':
            subdataset = datamodule.test_set
        else:
            raise ValueError(f'{subdataset_name} option not available.')

        for ckpt in get_checkpoints(FLAGS.rootdir, log_name=None, deep=True, metric_name='val_mse', sort_order='min'):
            print(f'Working on {ckpt}.')

            log_dict = evaluate_ckpt(class_=class_, dataset=subdataset, ckpt=ckpt,
                                     is_graph=FLAGS.model_class != 'SingleSiteMLP')
            log_dict['subdataset'] = subdataset_name
            df_logging.append(log_dict)

    df_logging = pd.DataFrame(df_logging)
    cols = [x for x in df_logging.columns if x not in log_dict]
    df_logging[cols] = df_logging[cols].dropna(axis=1, how='all')
    df_logging.to_csv(f'{dataset_name}/eval_{FLAGS.model_class}.csv')


if __name__ == '__main__':
    app.run(main)
