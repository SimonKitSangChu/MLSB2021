from absl import app, flags
import pandas as pd

from proteingnn.data import *
from proteingnn.example.data import EmbLibraryCreator, read_DeepSequence_csv


FLAGS = flags.FLAGS
flags.DEFINE_integer('n_processes', 1, 'Number of processes for dataset building.')
flags.DEFINE_multi_integer('radii', 6, 'Radii for edge definition.')
flags.DEFINE_boolean('embedding_only', False, 'Skip graph dataset creation.')
flags.DEFINE_multi_string('embedding', 'esm', 'Embedding options (esm/protbert/pssm/onehot)')
flags.DEFINE_float('radial_sigma', 0., 'Normal noise on atom cartesian pos.')
flags.DEFINE_boolean('all_esm_layers', False, 'Generate ESM embedding on all layers for mutational effect propagation '
                                              'analysis')
flags.DEFINE_string('embedding_model_name', None, 'ESM/ProtBert model to be used')
flags.DEFINE_integer('n_pdbs', 1, 'Number of relaxed structures used from alphafold2.')
flags.DEFINE_multi_string('dataset', None, 'Dataset(s) to generate. (Default: all datasets)')
flags.DEFINE_boolean('drop_dummy', True, 'Drop dummy mutations in dataset(s), such as A111A.')


def main(argv):
    pyrosetta.init('-mute all')

    embedding_dir = Path('data/embeddings')
    if not embedding_dir.exists():
        embedding_dir.mkdir()

    af_dir = Path('data/alphafold2')
    if not FLAGS.embedding_only and not af_dir.exists():
        raise FileNotFoundError(f'alphafold2 folder not found for graph dataset building.')

    mt_struct_dir = Path('data/mutant_structures')
    if not mt_struct_dir.exists():
        mt_struct_dir.mkdir()

    csv_dir = Path('data/csv')
    csv_dict = {csv.stem: csv for csv in csv_dir.glob('*.csv')}

    fasta_dir = Path('data/fasta')

    for fasta in fasta_dir.glob('*.fasta'):
        dataset_name = fasta.stem
        pdb_code = dataset_name.replace('_', '')

        # skip irrelevant datasets
        if FLAGS.dataset is not None and dataset_name not in FLAGS.dataset:
            continue

        print(f'Proceed to {fasta}')

        af_subdir = af_dir / dataset_name
        mt_struct_subdir = mt_struct_dir / dataset_name

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
        if FLAGS.drop_dummy:
            df = df.loc[[i for i in df.index if i not in ('WT', 'wt') and i[0] != i[-1]]]
        else:
            df = df.loc[[i for i in df.index if i not in ('WT', 'wt')]]
        exp_data = df['exp'].dropna().to_dict()

        # create embedding library
        emb_creator = EmbLibraryCreator(rootdir=embedding_dir / dataset_name, fasta=fasta, exp_data=exp_data)
        try:
            if 'esm' in FLAGS.embedding:
                emb_creator.create_embedding_library(embedding_name='esm', pdb_code=pdb_code, pssm_dim=None, use_diff=True,
                                                     model_name=FLAGS.embedding_model_name,
                                                     layers=list(range(34)) if FLAGS.all_esm_layers else [33])
            if 'protbert' in FLAGS.embedding:
                emb_creator.create_embedding_library(embedding_name='protbert', pdb_code=pdb_code, pssm_dim=None,
                                                     use_diff=True, model_name=FLAGS.embedding_model_name)
            if 'pssm' in FLAGS.embedding:
                emb_creator.create_embedding_library(embedding_name='pssm', pdb_code=pdb_code, pssm_dim=1)
            if 'onehot' in FLAGS.embedding:
                emb_creator.create_embedding_library(embedding_name='onehot', pdb_code=pdb_code, pssm_dim=None, use_diff=True)

        except (ValueError, AssertionError) as e:
            print(e)
            continue

        if FLAGS.embedding_only:
            continue

        # generate dummy mutants (assume static wildtype / mutant structure)
        pdbs = sorted(af_subdir.glob('ranked_*.pdb'))
        if not pdbs:
            warnings.warn(f'No ranked relaxed alphafold structure found in {af_subdir}.')

        for pdb in pdbs[:FLAGS.n_pdbs]:
            emb_creator.create_mutant_pdbs(
                pdb_code=pdb_code,
                src_dir=af_subdir,
                dst_dir=mt_struct_subdir,
                pdb=pdb.name
            )

        # generate graph dataset
        n_processes = FLAGS.n_processes

        for radius in FLAGS.radii:
            # ESM dataset
            fa_factory = DatasetFactory(
                name='ESMDatasetFactory',
                predataset_path=mt_struct_subdir,
                mutant_y=exp_data
            )
            fa_factory.node_filter = AtomNameNodeFilter(atom_name_pass=['CA'])
            fa_factory.edge_featurizer = DistanceEdgeFeaturizer(max_distance=radius, sigma=FLAGS.radial_sigma,
                                                                is_edge_only=True)

            if 'esm' in FLAGS.embedding:
                fa_factory.node_featurizer = SeqEmbNodeFeaturizer(emb_dir=embedding_dir / dataset_name / 'esm')
                fa_factory.dataset_path = Path(f'datasets/{dataset_name}/esm-{radius}')
                fa_factory.dump_config(overwrite=True)
                fa_factory.create_dataset(n_processes=n_processes, pos_flag=True)

            # onehot dataset
            if 'onehot' in FLAGS.embedding:
                fa_factory.name = 'OnehotDatasetFactory'
                fa_factory.node_featurizer = SeqEmbNodeFeaturizer(emb_dir=embedding_dir / dataset_name / 'onehot')
                fa_factory.dataset_path = Path(f'datasets/{dataset_name}/onehot-{radius}')
                fa_factory.dump_config(overwrite=True)
                fa_factory.create_dataset(n_processes=n_processes, pos_flag=True)

            # (1D-)PSSM dataset
            if 'pssm' in FLAGS.embedding:
                # warning: not tested
                fa_factory.name = 'PSSMDatasetFactory'
                fa_factory.node_featurizer = SeqEmbNodeFeaturizer(emb_dir=embedding_dir / dataset_name / 'pssm')
                fa_factory.dataset_path = Path(f'datasets/{dataset_name}/1Dpssm-{radius}')
                fa_factory.dump_config(overwrite=True)
                fa_factory.create_dataset(n_processes=n_processes, pos_flag=True)


if __name__ == '__main__':
    app.run(main)
