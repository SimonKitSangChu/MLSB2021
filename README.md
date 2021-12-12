# MLSB2021

<!-- Geometric architecture visualization --> 
This project requires [ProteinGNN](https://github.com/SimonKitSangChu/ProteinGNN) to parse pdb to PyG
compactible format. Please follow the installation process there.

To build the datasets, place all AlphaFold2 structures under `data/alphafold2/your_dataset`, fasta under `data/fasta`
and experiment csv under `data/csv`.
```
python build_dataset.py --embedding esm --radii 6 --dataset your_dataset --n_processes N_PROCESSES
```

To train sequence-only and geometric models and visualize their performances,
```
bash batch_train.sh
python compare_supervised.py --rootdir esm-6
```
<!-- [](imgs/supervised.png) -->

To further compare with unsupervised predictions, place pssm files under `data/pssm` and run
```
python ESM.py --stage preprocess --esm_install_path ESM_INSTALL_DIR
bash benchmark_esm.sh
python ESM.py --stage postprocess
python compare_unsupervised.py
```
<!-- [](imgs/unsupervised.png) -->

For embedding locality analysis, 
```
python locality.py
```
