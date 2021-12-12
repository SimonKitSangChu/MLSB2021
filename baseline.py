from absl import app
from absl import flags
from collections import OrderedDict
import itertools
import pandas as pd
from pathlib import Path
import pickle
import sys

from proteingnn.model import get_default_trainer, DummyGraphConv
from proteingnn.data import BaseDatamodule
from proteingnn.example.data import SingleSiteDataset, read_DeepSequence_csv, DefaultDatamodule
from proteingnn.example.model import SingleSiteMLP, FastMLPModel, FastGCNModel, SeqPoolingMLP

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'Run one batch on CPU before training.')
flags.DEFINE_boolean('verbose', False, 'Verbose mode')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_boolean('reboot', True, 'Restart from parameter scanning.')
flags.DEFINE_integer('model_repeat', 1, 'Number of training ensembles for each parameter.')
flags.DEFINE_multi_integer('hidden_channels', 16, 'Hidden layer size.')
flags.DEFINE_multi_float('weight_decay', 0., 'L2 regularization through optimizer.')
flags.DEFINE_integer('patience', 10, 'Patience in training.')
flags.DEFINE_multi_string('model_name', None, 'Options: SingleSiteMLP, FastMLP, SeqPoolingMLP, DummyGraphConv, all')
flags.DEFINE_bool('regression', True, 'regression/classification')
flags.DEFINE_string('dataset_name', None, 'Dataset name.')
flags.DEFINE_string('embedding_radius', 'esm-6', '{Embedding name}-{radius}')
flags.DEFINE_integer('n_lin_layers', 3, 'Number of linear layers.')
flags.DEFINE_integer('n_gnn_layers', 3, 'Number of gnn layers.')


def main(argv):
    dataset_name = FLAGS.dataset_name
    embeddding_name, radius = FLAGS.embedding_radius.split('-')
    pdb_code = dataset_name.replace('_', '')

    # read dataset
    csv_dir = Path('data/csv')
    csv_dict = {csv.stem: csv for csv in csv_dir.glob('*.csv')}

    df = pd.read_csv(csv_dict[dataset_name])
    df = df.set_index('mutant')
    df = df.loc[[i for i in df.index if i not in ('WT', 'wt')]]
    exp_data = df['exp'].dropna().to_dict()

    # parameter grid
    common_grid = OrderedDict([
        ('hidden_channels', FLAGS.hidden_channels),
        ('weight_decay', FLAGS.weight_decay),
        ('n_lin_layers', (FLAGS.n_lin_layers,)),
        ('norm_name', (None,)),
        ('lr', (5e-3,))
    ])

    # save common grid progress
    grid_pkl = Path('common_grid_baseline.pkl')
    if grid_pkl.exists() and FLAGS.reboot:
        grid = pickle.load(grid_pkl.open('rb'))
        if not grid:
            raise ValueError(f'{grid_pkl} is empty.')
    else:
        common_grid_keys = list(common_grid)
        grid = []

        for param in itertools.product(*common_grid.values()):
            dic = {common_grid_keys[i]: v for i, v in enumerate(param)}
            grid.append(dic)

        pickle.dump(grid, grid_pkl.open('wb'))

    # avoid re-running
    for model_name in ('SingleSiteMLP', 'FastMLP', 'SeqPoolingMLP', 'DummyGraphConv'):
        if model_name in FLAGS.model_name or FLAGS.model_name[0] == 'all':
            logdir = Path(f'{dataset_name}/{model_name}')
            if logdir.exists():
                raise FileExistsError(f'{logdir} already exists. Skip re-running now.')

    while grid:
        common_param = grid.pop()

        # model training
        for i_repeat in range(1, FLAGS.model_repeat + 1):
            #### SingleSiteMLP
            if 'SingleSiteMLP' in FLAGS.model_name or 'all' in FLAGS.model_name:
                # single site dataset
                dataset = SingleSiteDataset(
                    root=f'data/embeddings/{dataset_name}/{embeddding_name}',
                    exp_data=exp_data,
                    pdb_code=pdb_code,
                )
                datamodule = BaseDatamodule(
                    dataset=dataset, batch_size=FLAGS.batch_size, pin_memory=True, num_workers=1,
                    split_root=f'datasets/{dataset_name}'
                )

                x, _ = datamodule.example_input_array
                in_channels = x.shape[1]
                model = SingleSiteMLP(in_channels=in_channels, regression=FLAGS.regression, **common_param)
                _ = model.forward(x)  # dry-run for LazyModule

                # training
                print()
                print(f'##### SingleSiteMLP repeat {i_repeat} #####')
                print(common_param)
                print()

                trainer = get_default_trainer(logdir=dataset_name, log_name='SingleSiteMLP', gpus=[0], restart=False,
                                              debug=FLAGS.debug, patience=FLAGS.patience)
                trainer.fit(model, datamodule)

            if len(FLAGS.model_name) == 1 and FLAGS.model_name == 'SingleSiteMLP':
                continue

            # graph dataset
            datamodule = DefaultDatamodule(
                root=f'datasets/{dataset_name}/{embeddding_name}-{radius}', exp_data=exp_data,
                batch_size=FLAGS.batch_size, pin_memory=True, num_workers=1,
                split_root=f'datasets/{dataset_name}'
            )
            x = datamodule.example_input_array
            in_channels = x.x.shape[1]
            num_nodes = x.x.shape[0] // FLAGS.batch_size  # warning: assume uniform structure size

            #### FastMLP
            if 'FastMLP' in FLAGS.model_name or 'all' in FLAGS.model_name:
                model = FastMLPModel(num_nodes=num_nodes, in_channels=in_channels,
                                     regression=FLAGS.regression, **common_param)

                _ = model.forward(x.x)  # dry-run for LazyModule

                # training
                print()
                print(f'##### FastMLP repeat {i_repeat} #####')
                print(common_param)
                print()

                trainer = get_default_trainer(logdir=dataset_name, log_name='FastMLP', gpus=[0], restart=False,
                                              debug=FLAGS.debug, patience=FLAGS.patience)
                trainer.fit(model, datamodule)

            #### SeqPoolingMLP (duplicate of DummyGraphConv)
            if 'SeqPoolingMLP' in FLAGS.model_name or 'all' in FLAGS.model_name:
                param = common_param.copy()
                del param['n_lin_layers']
                model = SeqPoolingMLP(n_layers=common_param['n_lin_layers'], in_channels=in_channels,
                                      regression=FLAGS.regression, **param)

                x = datamodule.example_input_array
                _ = model.forward(x.x, x.edge_index, x.batch)  # dry-run for LazyModule

                # training
                print()
                print(f'##### SeqPoolingMLP repeat {i_repeat} #####')
                print(common_param)
                print()

                trainer = get_default_trainer(logdir=dataset_name, log_name='SeqPoolingMLP', gpus=[0], restart=False,
                                              debug=FLAGS.debug, patience=FLAGS.patience)
                trainer.fit(model, datamodule)

            #### DummyGraphConv
            if 'DummyGraphConv' in FLAGS.model_name or 'all' in FLAGS.model_name:
                param = common_param.copy()
                del param['norm_name']

                model = FastGCNModel(
                    gnn_class=DummyGraphConv,
                    in_channels=in_channels,
                    n_gnn_layers=3,
                    gnorm_class=None,  # to mimic MLP
                    norm_name=None,
                    regression=FLAGS.regression,
                    **param
                )

                x = datamodule.example_input_array
                _ = model.forward(x.x, x.edge_index, x.batch)  # dry-run for LazyModule

                # training
                print()
                print(f'##### DummyGraphConv repeat {i_repeat} #####')
                print(common_param)
                print()

                trainer = get_default_trainer(logdir=dataset_name, log_name='DummyGraphConv', gpus=[0], restart=False,
                                              debug=FLAGS.debug, patience=FLAGS.patience)
                trainer.fit(model, datamodule)

        pickle.dump(grid, grid_pkl.open('wb'))

    if not grid:
        if grid_pkl.exists():
            grid_pkl.unlink()


if __name__ == '__main__':
    app.run(main)
