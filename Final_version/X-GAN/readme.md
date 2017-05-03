# X-GAN Instructions

To train this XGAN model and see the inpainting ability of generator, following below commands:

#### 0.You need to prepare 2 dataset, one is original images and the other is noised original images, no need to have same images

#### 1.Train XGAN

If dataset (all images under `./data/xxx1` and `./data/xxx2`)is already resized to 64x64, run:

`python main.py --dataset xxx1 --is_train --is_crop False --epoch 100 --dataset2 xxx2`

Otherwise, run:

`python main.py --dataset xxx1 --is_train --is_crop True --epoch 100 --dataset2 xxx2`

#### 2.Use model as an inpainter

`imgs`: path to testing dataset

`outDir`: path to where to output inpainted results

`maskType`: mask types :['center', 'random', 'left', 'right']

`nIter`: inpainting iterations

`checkpointDir`: path to trained model checkpoint directory


If the same datasets as training process (all images under `./data/xxx1` and `./data/xxx2` and testing data `xxx_test`) are already resized to 64x64, run:

`python inpainter.py --dataset xxx1 --dataset2 xxx2 --imgs xxx_test --outDir path_to_where_to_output_inpainted_results --is_train --nIter 2000 --is_crop False --maskType center --checkpointDir path_to_trained_model_checkpoint_directory`

Otherwise, run:

`python inpainter.py --dataset xxx1 --dataset2 xxx2 --imgs data/catface_O_r --outDir path_to_where_to_output_inpainted_results --is_train --nIter 2000 --is_crop True --maskType center --checkpointDir path_to_trained_model_checkpoint_directory`


#### References
1. Yeh, Raymond, et al. "Semantic Image Inpainting with Perceptual and Contextual Losses." arXiv preprint arXiv:1607.07539 (2016).
2. Arjovsky, Martin, Soumith Chintala, and Léon Bottou. "Wasserstein gan." arXiv preprint arXiv:1701.07875 (2017).
3. Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
4. Goodfellow, Ian, et al. "Generative adversarial nets." Advances in Neural Information Processing Systems. 2014.

##### Note: You can use ImageMagick to generate GIFs from resulted images

    
