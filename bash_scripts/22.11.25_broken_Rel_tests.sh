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
  # configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V6.yaml

  # using W_edge but with A(1) not A_e -- one step away from original Rel-r*GCN
  # configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V7.yaml

  # using W_kt with e parts, a la V6, but using A_e now -- this should be a finished version, if it works (still no idea what's wrong w V7 but oh well?)
  configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V8.yaml

  # hopefully the finished, fixed version. The problem was that we were sharing weights across t for the first hop; just having W_e, akin to delite_gcn
    configs/debug_cfgs/peptides-func-RelDelayGCN_L=05_d=064_rbar=inf_Rel_V9.yaml
)
for run in "${runs[@]}" ; do
  # python main.py --cfg "$run" --repeat 3 device cuda dataset.dir /data/beng/datasets train.batch_size 16
  python main.py --cfg "$run" --repeat 1 device cuda dataset.dir datasets out_dir "results/$BATCH" optim.max_epoch 50
  python bash_scripts/progress_bar.py "$run"
done