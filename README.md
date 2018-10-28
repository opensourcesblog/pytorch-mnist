# pytorch-mnist
## Pytorch MNIST and preprocessing


   
## mnist.py
 `SUCCESS` will be in the form of something like this:
> Test set: Average loss: 0.0300, Accuracy: 4/4 (100%)  

You can train the model yourself by `python mnist.py --train` and you can switch off pretrained as well:
`python mnist.py --train --no_pretrained`
 
## predict_interface_usage.py
 There is a test image in `img/` with the name `test_2.png` so you can run:
 `python predict_interface_usage.py test_2` to predict the handwritten digits.

 `SUCCESS` will write an output with predictions to the command
 prompt and it will generate an image with the predictions in `/pro-img/`
