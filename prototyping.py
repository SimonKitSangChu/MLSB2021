from absl import app
from absl import flags
from collections import OrderedDict
import itertools
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import ParameterGrid

from proteingnn.model import get_default_trainer
from proteingnn.example.data import DefaultDatamodule, read_DeepSequence_csv
from proteingnn.example.model import FastGCNModel

FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'Run one batch on CPU before training.')
flags.DEFINE_multi_string('datasets', None, 'Dataset(s) use`d in training.')
flags.DEFINE_multi_integer('hidden_channels', 16, 'Hidden layer size.')
flags.DEFINE_boolean('verbose', False, 'Verbose mode')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_string('reboot_pkl', None, 'Restart from parameter scanning.')
flags.DEFINE_integer('model_repeat', 1, 'Number of training ensembles for each parameter.')
flags.DEFINE_multi_string('gnn_name', ['GCNConv'], 'GNN name in torch_geometric.nn')
flags.DEFINE_integer('patience', 50, 'Patience in training.')
flags.DEFINE_multi_float('weight_decay', 0, 'Weight decay in Adam.')
flags.DEFINE_bool('regression', True, 'regression/classification')
flags.DEFINE_string('dataset_name', None, 'Dataset name.')
flags.DEFINE_string('embedding_radius', 'esm-6', '{Embedding name}-{radius}')
flags.mark_flags_as_required(['dataset_name'])


def main(argv):
    dataset_name = FLAGS.dataset_name
    embeddding_name, radius = FLAGS.embedding_radius.split('-')

    common_grid = OrderedDict([
        ('hidden_channels', FLAGS.hidden_channels),
        ('weight_decay', FLAGS.weight_decay),
        ('lr', (5e-3,)),
    ])

    # read dataset
    csv_dir = Path('data/csv')
    csv_dict = {csv.stem: csv for csv in csv_dir.glob('*.csv')}

    df = pd.read_csv(csv_dict[dataset_name])
    df = df.set_index('mutant')
    df = df.loc[[i for i in df.index if i not in ('WT', 'wt')]]
    exp_data = df['exp'].dropna().to_dict()

    # avoid rerunning
    for model_name in FLAGS.gnn_name:
        logdir = Path(f'{dataset_name}/{model_name}')
        if logdir.exists():
            raise FileExistsError(f'{logdir} already exists. Skip re-running now.')

    # parameter grid
    model_grids = {
        'GCNConv': {
        },
        'ChebConv': {
            'K': (4,),
        },
        'SAGEConv': {
        },
        'GraphConv': {
            'aggr': ('mean',),
        },
        'ResGatedGraphConv': {
        },
        'ResGraphConv': {
        },
        'GATConv': {
            'heads': (1, 2),
            'concat': (False,),
            'negative_slope': (0.2,),
            'dropout': (0.3,),
        },
        'GATv2Conv': {
            'heads': (1, 2),
            'concat': (False,),
            'negative_slope': (0.2,),
            'dropout': (0.3,),
            'share_weights': (True, False),
        },
        'TransformerConv': {
            'heads': (1, 2),
            'concat': (False,),
            'beta': (True, False,),
            'dropout': (0.3,),
        },
    }
    model_grids = {key: model_grids[key] for key in FLAGS.gnn_name}

    # save common grid progress
    if FLAGS.reboot_pkl:
        grid_pkl = Path(FLAGS.reboot_pkl)
        grid = pickle.load(grid_pkl.open('rb'))
    else:
        # identified non-crashing pkl name
        i = 0
        grid_pkl = Path(f'common_grid_{i}.pkl')
        while grid_pkl.exists():
            i += 1
            grid_pkl = Path(f'common_grid_{i}.pkl')

        # create and save common grid
        common_grid_keys = list(common_grid)
        grid = []

        for param in itertools.product(*common_grid.values()):
            dic = {common_grid_keys[i]: v for i, v in enumerate(param)}
            grid.append(dic)

        pickle.dump(grid, grid_pkl.open('wb'))

    while grid:
        common_param = grid.pop()

        datamodule = DefaultDatamodule(
            root=f'datasets/{dataset_name}/{embeddding_name}-{radius}', exp_data=exp_data,
            batch_size=FLAGS.batch_size, pin_memory=True, num_workers=1,
            split_root=f'datasets/{dataset_name}'
        )
        if FLAGS.verbose:
            datamodule.print_summary()

        for model_name, model_grid in model_grids.items():
            for param_dict in ParameterGrid(model_grid):
                for i_repeat in range(1, FLAGS.model_repeat+1):
                    model = FastGCNModel(
                        regression=FLAGS.regression,
                        n_gnn_layers=3,
                        n_lin_layers=3,
                        gnn_class=model_name,
                        gnorm_class=None,
                        norm_name=None,
                        in_channels=datamodule.num_node_features,
                        leakyrelu_slope=0.2,
                        batch_size=FLAGS.batch_size,
                        **common_param,
                        **param_dict
                    )

                    input = datamodule.example_input_array.clone()
                    _ = model.forward(input.x, input.edge_index, input.batch)

                    print()
                    print(f'##### {model_name} repeat {i_repeat} #####')
                    print(dataset_name)
                    print(model.hparams, param_dict)
                    print()

                    # cool down period of 50 epochs follow by patience
                    trainer = get_default_trainer(logdir=dataset_name, log_name=model_name, gpus=[0], restart=False,
                                                  debug=FLAGS.debug, patience=50, max_epochs=50)
                    trainer.fit(model, datamodule)

        pickle.dump(grid, grid_pkl.open('wb'))

    if not grid:
        if grid_pkl.exists():
            grid_pkl.unlink()


if __name__ == '__main__':
    app.run(main)
