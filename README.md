# RecoverGAN Final Version: X-GAN

## Overview

* Structure


<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/Structure.png?raw=true" width="40%"/>
</p>

* Some results

1. Inpainting center cropped cat faces

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/cat_before.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/cat_masked.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/inpaint_cat_xcsGAN_300-2000_center.gif?raw=true" width="70%"/>
</p>

2. Inpainting random noised cat faces

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/cat_before.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/cat_masked_random.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/inpaint_cat_xcsGAN_300-2000_random.gif?raw=true" width="70%"/>
</p>

3. Inpainting center cropped aircrafts

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/aircraft_before.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/aircraft_masked.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/inpaint_plane_xcsGAN_21-2000_center.gif?raw=true" width="70%"/>
</p>

4. Inpainting random noised aircrafts

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/aircraft_before.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/aircraft_masked_random.png?raw=true" width="70%"/>
</p>

<p align="center">
  <img src="https://github.com/rexwangcc/RecoverGAN/blob/master/Final_version/results/inpaint_plane_xcsGAN_21-2000_random.gif?raw=true" width="70%"/>
</p>

### Advantages

* Has decent inpainting performance on cats and planes
* Robust to different noises at random locations
* Our model learns much faster than original GAN
* Based on DCGAN, Context Encoder and WGAN, open to many extensions

### Open-Challenges and future works

* Further modify the inputs for generator / discriminator:
    + Compressive sensing technique (maybe a compressive sensing layer)
    + An extra encoder to encode noisy images
    + Additional inputs (different types of noise)
* Wider applications for generators (inpainters):
    + Wider varieties of categories of images
    + Wider areas (Music, Video, Creative Works, etc.)
* Modify the network:
    + Stacked GANs
    + Add dropouts
    + Optimized formulas


### References
1. Yeh, Raymond, et al. "Semantic Image Inpainting with Perceptual and Contextual Losses." arXiv preprint arXiv:1607.07539 (2016).
2. Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein gan." arXiv preprint arXiv:1701.07875 (2017).
3. Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
4. Goodfellow, Ian, et al. "Generative adversarial nets." Advances in Neural Information Processing Systems. 2014.




## Instructions

#### You need to prepare 2 dataset, one is original images and the other is noised original images, no need to have same images

#### Train XGAN

If dataset (all images under `./data/xxx1` and `./data/xxx2`)is already resized to 64x64, run:

`python main.py --dataset xxx1 --is_train --is_crop False --epoch 100 --dataset2 xxx2`

Otherwise, run:

`python main.py --dataset xxx1 --is_train --is_crop True --epoch 100 --dataset2 xxx2`

#### Use model as an inpainter

`imgs`: path to testing dataset

`outDir`: path to where to output inpainted results

`maskType`: mask types :['center', 'random', 'left', 'right']

`nIter`: inpainting iterations

`checkpointDir`: path to trained model checkpoint directory


If the same datasets as training process (all images under `./data/xxx1` and `./data/xxx2` and testing data `xxx_test`) are already resized to 64x64, run:

`python inpainter.py --dataset xxx1 --dataset2 xxx2 --imgs xxx_test --outDir path_to_where_to_output_inpainted_results --is_train --nIter 2000 --is_crop False --maskType center --checkpointDir path_to_trained_model_checkpoint_directory`

Otherwise, run:

`python inpainter.py --dataset xxx1 --dataset2 xxx2 --imgs data/catface_O_r --outDir path_to_where_to_output_inpainted_results --is_train --nIter 2000 --is_crop True --maskType center --checkpointDir path_to_trained_model_checkpoint_directory`



