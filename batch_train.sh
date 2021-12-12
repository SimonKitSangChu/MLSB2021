#!/bin/bash

repeat=5
patience=25
er='esm-6'
batch_size=256

if [ ! -d $er ]; then
    mkdir $er
fi

for dataset in `ls datasets`; do
    python baseline.py --embedding_radius $er --dataset_name $dataset --model_repeat $repeat \
        --patience $patience --batch_size $batch_size --hidden_channels 32 --weight_decay 1e-4 --n_lin_layers 3 \
        --model_name SingleSiteMLP

   python baseline.py --embedding_radius $er --dataset_name $dataset --model_repeat $repeat \
        --patience $patience --batch_size $batch_size --hidden_channels 32 --weight_decay 0 --n_lin_layers 3 \
        --model_name DummyGraphConv

    python baseline.py --embedding_radius $er --dataset_name $dataset --model_repeat $repeat \
        --patience $patience --batch_size $batch_size --hidden_channels 32 --weight_decay 0 --n_lin_layers 3 \
        --model_name SeqPoolingMLP

    python prototyping.py --embedding_radius $er --dataset_name $dataset --model_repeat $repeat \
        --patience $patience --batch_size $batch_size --hidden_channels 32 --weight_decay 0 \
        --gnn_name GCNConv

    for model_class in 'SingleSiteMLP' 'GCNConv' 'DummyGraphConv' 'SeqPoolingMLP'; do
        python batch_eval.py --rootdir $dataset/$model_class --dataset $dataset --subdataset val --subdataset test --embedding_radius $er --model_class $model_class
    done

    mv $dataset $er/
done

python compare_supervised.py --rootdir $er

