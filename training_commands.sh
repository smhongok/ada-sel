# brain_kmeans_3 for acceleration 8
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_kmeans_1of3 --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/multibrain_vmap_kmeans/vmap_kmeans_1of3.npy
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_kmeans_2of3 --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/multibrain_vmap_kmeans/vmap_kmeans_2of3.npy
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_kmeans_3of3 --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/multibrain_vmap_kmeans/vmap_kmeans_3of3.npy

# evaluation for validation dataset
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_kmeans_1of3 --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/multibrain_vmap_kmeans/vmap_kmeans_1of3.npy --mode test
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_kmeans_2of3 --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/multibrain_vmap_kmeans/vmap_kmeans_2of3.npy --mode test
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_kmeans_3of3 --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/multibrain_vmap_kmeans/vmap_kmeans_3of3.npy --mode test


## Baseline
# brain_variable_density for acceleration 8
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_vd --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/vd_probs_npy/vd_prob_acc8.npy

# evaluation for validation dataset
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_vd --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/vd_probs_npy/vd_prob_acc8.npy --mode test

# brain_random_mask for acceleration 8
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_random --gpu 1 --batch_size 4 --accelerations 8 --mask_type random

# evaluation for validation dataset
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_random --gpu 1 --batch_size 4 --accelerations 8 --mask_type random --mode test