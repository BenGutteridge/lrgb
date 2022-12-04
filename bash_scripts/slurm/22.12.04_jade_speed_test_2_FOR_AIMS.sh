#!/bin/bash
cd ../..
DIR_NAME="jade_speed_test_2"
python bash_scripts/progress_bar.py
runs=(
configs/rbar-GCN/rbar=inf/peptides-func-DelayGCN_L=05_d=175_rbar=inf.yaml
)
batch_sizes=(
  128
  # 256
)
hidden_dims=(
  # 64
  64
)
num_layers=(
  # 6
  5
  7
  9
  11
  13
  15
  17
  19
  21
  23
  25
)

L="$1"

echo "START"
current_date_time="`date +%Y%m%d%H%M%S`";
echo $current_date_time;

for L in "${num_layers[@]}" ; do
  for d in "${hidden_dims[@]}" ; do
    for bs in "${batch_sizes[@]}" ; do
      DIR="results/$DIR_NAME\_bs=$bs\_d=$d\_L=$L"
      mkdir -p DIR
      for run in "${runs[@]}" ; do
        echo "RUN: $DIR"
        # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
        python3 main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets out_dir "$DIR" optim.max_epoch 1 train.batch_size "$bs" gnn.dim_inner "$d" gnn.layers_mp "$L"
        python3 bash_scripts/progress_bar.py "$run"
      done
    done
  done
done

echo "END"