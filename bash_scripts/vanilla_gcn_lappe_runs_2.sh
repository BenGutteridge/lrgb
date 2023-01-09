
#!/bin/bash
python main.py --cfg configs/GCN/peptides-struct-GCN+LapPE.yaml --repeat 3 gnn.layers_mp 5 gnn.dim_inner 300
# python main.py --cfg configs/GCN/peptides-struct-GCN+LapPE.yaml --repeat 3 gnn.layers_mp 9 gnn.dim_inner 230
# python main.py --cfg configs/GCN/peptides-struct-GCN+LapPE.yaml --repeat 3 gnn.layers_mp 13 gnn.dim_inner 190
python main.py --cfg configs/GCN/peptides-struct-GCN+LapPE.yaml --repeat 3 gnn.layers_mp 17 gnn.dim_inner 170