########################
Prerequisites:
    tensorflow 1.0
    python 3.6
    pillow
    scipy

Datasets:
    10 K cats and their blurred version

Related works:
    https://github.com/carpedm20/DCGAN-tensorflow
########################


# Example, standard
python3 main.py --dataset catface --dataset2 catblur --is_crop --is_train --epoch 600

# Example, scc version:
qsub -l gpus=1 -l gpu_c=3.5 -j y -P dlearn run.sh


# To train this DCGAN-xGAN model v1, following below commands:

### Train DCGAN
## You need to prepare 2 dataset, one is original image and the other is noised original images
If dataset(all images under `./data/xxx1` and `./data/xxx2`)is already resized to 64x64, run:
`python main.py --dataset xxx1 --is_train --is_crop False --epoch 100 --dataset2 xxx2`

Otherwise, run:

`python main.py --dataset xxx1 --is_train --is_crop True --epoch 100 --dataset2 xxx2`
