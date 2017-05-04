## To train this DCGAN-SGAN model and see the inpainting ability of generator, following below commands:

### 1.Train DCGAN

If dataset(all images under `./data/xxx`)is already resized to 64x64, run:
`python main.py --dataset xxx --is_train --is_crop False --epoch 1000 `

Otherwise, run:
`python main.py --dataset xxx --is_train --is_crop True --epoch 1000 `

### 2.Use model as an inpainter

`imgs`: path to testing dataset
`outDir`: path to where to output inpainted results
`maskType`: mask types :['center', 'random', 'left', 'right']

If the same dataset as training process(all images under `./data/xxx` and testing data)is already resized to 64x64, run:
`python inpainter.py --nIter 10000 --is_crop False --dataset xxx --imgs path_to_testing_dataset --outDir path_to_where_to_output_inpainted_results --maskType center`

Otherwise, run:
`python inpainter.py --nIter 10000 --is_crop True --dataset xxx --imgs path_to_testing_dataset --outDir path_to_where_to_output_inpainted_results --maskType center`


### This implementation is based on the paper: Yeh, Raymond, et al. "Semantic Image Inpainting with Perceptual and Contextual Losses." arXiv preprint arXiv:1607.07539 (2016).

    
