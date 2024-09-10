# [ECCV 2024] Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing

Official repository of a paper 'Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing' 

by [Seongmin Hong](https://smhongok.github.io), [Jaehyeok Bae](https://jaehyeokbae.me/), [Jongho Lee](https://list.snu.ac.kr/jongho-lee), and [Se Young Chun](https://icl.snu.ac.kr/pi).
 

We provide codes, divided into three parts:
I. Training / Inference of SR space generation model [./flow](./flow)
II. Generation of k-means++ centroid of HF Bayesian uncertainty (c_j)
III. Training / Inference of Reconstruction network

## I. Training / Inference of SR space generation model

The [./flow](./flow) directory contains the code for flow-based SR space generation, which generates HF Bayesian uncertainty. The file [./flow/create_dataset.ipynb](./flow/create_dataset.ipynb) generates LR (low-resolution) and HR (high-resolution) images, which serve as input and target images for flow-based SR space generation, respectively, from fastMRI data in .h5 files. The options for [./flow/train.py](./flow/train.py) are adjusted in the [./flow/confs](./flow/confs) folder. To run 'train.py', use the following command:

```bash
cd ./flow
python train.py -opt confs/8x_mr_server.yml
```

To test and generate HF Bayesian uncertainty, use the following command:
```bash
python test_with_vmap.py --conf_path experiments/YOUR-OWN-PATH/config.yml --lrtest_path YOUR-OWN-PATH/LR/ --output_path YOUR-OWN-OUTPUT-PATH --n_imgs N-IMGS 
```

In order for running this code, you should check the original repository of FS-NCSR [Github](https://github.com/dsshim0125/FS-NCSR) and install the environment of it.

## II. Generation of k-means++ centroid of HF Bayesian uncertainty (c_j)

Prepare the HF Bayesian uncertainty maps of the training data. 
Then run [kmeans.ipynb](kmeans.ipynb).

## III. Training / Inference of Reconstruction network

### Environment
We trained and evaluated our models in Python 3.9 and PyTorch which version torch=1.12.1 and torchvision=0.13.1. Training is on NVIDIA GeForce RTX 3090Ti GPU setting. We used pytorch-lightning=1.0.4 framework to manage the training and evaluation process.

### Quick Guide
We provide a bash file for training and evaluating the reconstruction process with NYU fastMRI multi-coil brain dataset. The multi-coil brain dataset should be stored in [./dataset/multibrain_dataset](./dataset/multibrain_dataset) directory. Train set and validation set should be stored in [./dataset/multibrain_dataset/multicoil_train](./dataset/multibrain_dataset/multicoil_train) and [./dataset/multibrain_dataset/multicoil_val](./dataset/multibrain_dataset/multicoil_val), respectively.

 Additionally, probability maps for sampling are stored in [./dataset/multibrain_vmap_kmeans](./dataset/multibrain_vmap_kmeans) directory and [./dataset/vd_probs_npy](./dataset/vd_probs_npy) directory. [./dataset/multibrain_vmap_kmeans](./dataset/multibrain_vmap_kmeans) directory is for the proposed adaptive selection method, and [./dataset/vd_probs_npy](./dataset/vd_probs_npy) is for sampling from variable density.

First, go to the root repository and install all requirements of the environment. The environment name we used is "adaptive_selection".

```bash
conda create -y -n adaptive_selection
conda activate adaptive_selection 
```

Then, install directly from the source code.
```bash
pip install -e .
```

The available versions of torch and torchvision might be different depending on hardware settings.
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

Finally, downgrade the pytorch-lightning version to prevent errors.
```bash
pip install pytorch-lightning==1.0.4
```

Execute the provided bash file "training_commands.sh" for training and evaluating the proposed adaptive selection method for # of cluster = 3 and an acceleration rate of 8x. To compare with baselines, the bash file also contains the reconstruction model in which the mask is sampled from variable density or random.
```bash
chmod +x training_commands.sh
./training_commands.sh
```

As a result of execution, a "results" repository will be created. The structure of the repository is as follows. We provide the final execution result of the bash file in "./results" repository.

```
|-- brain_kmeans_1of3/
  |-- checkpoints/
  |-- final_output_imgs/
    file_brain_AXT2_200_2000078.h5_s0_error.png
    file_brain_AXT2_200_2000078.h5_s0_output.png
    file_brain_AXT2_200_2000078.h5_s0_target.png
    ...
  |-- lightning_logs/
  |-- masks/
    subsampling_mask.png
    vmap.png
  |-- metrics/
  metrics_final.csv
```

You can find the reconstructed image, ground truth(target), and error map for each slice of the dataset in "final_output_imgs" directory. You can also find the overall metrics of PSNR, SSIM, and NMSE on "metrics_final.csv". In "masks" directory, "vmap.png" refers to the probability map for sampling, and "subsampling_mask.png" is the corresponding sampling mask.

### Hyperparameter setting
The default training code for the proposed model is as below :
```bash
python ./fastmri_examples/advarnet2d/train_advarnet2d_demo.py --data_path ./dataset/multibrain_sample --default_root_dir ./results/brain_kmeans_1of3 --gpu 1 --batch_size 4 --accelerations 8 --vmap_target_path ./dataset/multibrain_vmap_kmeans/vmap_kmeans_1of3.npy
```
Note that the dataset should be in the directory of "--data_path" argument. You can change an acceleration rate with "--accelerations" argument, and the probability map with "--vmap_target_path" argument. To train the baseline with a random sampled mask, add "--mask_type random" argument. After the end of the training, the evaluation process can be executed by adding "--mode test" argument.


# Others

In this code, we only use T2 weighted images from the entire NYU fastMRI multi-coil brain dataset which contains 16 multi-coil for data consistency. To generate the same dataset, you can run "utils/volume_selection.ipynb".
In addition, you can run a k-means++ algorithm, using "utils/kmeanspp.ipynb".

## Reference
This code uses the fastMRI dataset from [the fastMRI dataset page](https://fastmri.med.nyu.edu/). Additionally, the code uses the facebookresearch fastMRI code from [GitHub](https://github.com/facebookresearch/fastMRI).

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/fastMRI/blob/master/LICENSE.md)

[Website and Leaderboards](https://fastMRI.org) |
[Dataset](https://fastmri.med.nyu.edu/) |
[GitHub](https://github.com/facebookresearch/fastMRI) |

For flow-based SR space generation, we modified FS-NCSR [Github](https://github.com/dsshim0125/FS-NCSR) and employed a model proposed by Hong et al. [arXiv](https://arxiv.org/abs/2212.04319)

## License

fastMRI is MIT licensed, as found in the [LICENSE file](https://github.com/facebookresearch/fastMRI/tree/master/LICENSE.md).

## Cite

```BibTeX
@inproceedings{zbontar2018fastMRI,
    title={{fastMRI}: An Open Dataset and Benchmarks for Accelerated {MRI}},
    author={Jure Zbontar and Florian Knoll and Anuroop Sriram and Tullie Murrell and Zhengnan Huang and Matthew J. Muckley and Aaron Defazio and Ruben Stern and Patricia Johnson and Mary Bruno and Marc Parente and Krzysztof J. Geras and Joe Katsnelson and Hersh Chandarana and Zizhao Zhang and Michal Drozdzal and Adriana Romero and Michael Rabbat and Pascal Vincent and Nafissa Yakubova and James Pinkerton and Duo Wang and Erich Owens and C. Lawrence Zitnick and Michael P. Recht and Daniel K. Sodickson and Yvonne W. Lui},
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1811.08839},
    year={2018}
}
@inproceedings{sriram2020endtoend,
    title={End-to-End Variational Networks for Accelerated MRI Reconstruction},
    author={Anuroop Sriram and Jure Zbontar and Tullie Murrell and Aaron Defazio and C. Lawrence Zitnick and Nafissa Yakubova and Florian Knoll and Patricia Johnson},
    year={2020},
    eprint={2004.06688},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
@article{bakker2022adaptive,
    title={On learning adaptive acquisition policies for undersampled multi-coil {MRI} reconstruction},
    author={Tim Bakker and Matthew Muckley and Adriana Romero-Soriano and Michal Drozdzal and Luis Pineda},
    journal={Proceedings of Machine Learning Research (MIDL)},
    pages={to appear},
    year={2022},
}
@inproceedings{song2022fs,
  title={FS-NCSR: Increasing Diversity of the Super-Resolution Space via Frequency Separation and Noise-Conditioned Normalizing Flow},
  author={Song, Ki-Ung and Shim, Dongseok and Kim, Kang-wook and Lee, Jae-young and Kim, Younggeun},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={967--976},
  year={2022},
  organization={IEEE}
}
@article{hong2022robustness,
  title={On the Robustness of Normalizing Flows for Inverse Problems in Imaging},
  author={Hong, Seongmin and Park, Inbum and Chun, Se Young},
  journal={ICCV},
  year={2023}
}
```
