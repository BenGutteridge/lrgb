#!/bin/bash
cd ..
BATCH="debugging_rel"
mkdir -p "results/$BATCH"
python bash_scripts/progress_bar.py
runs=(
  # configs/debug_cfgs/peptides-func-DelayGCN_L=05_d=064_rbar=inf.yaml
  # configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf.yaml

  # for v3 and v4 (run twice, but change which stage is activated for each)
  # configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V3.yaml
  # configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V4.yaml
  
  # building back up to R-r*GCN -- now including a summation
  # configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V5.yaml

  # now turning add_erdge_types back on
  configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V6.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir datasets out_dir "results/$BATCH" optim.max_epoch 50
  python bash_scripts/progress_bar.py "$run"
done