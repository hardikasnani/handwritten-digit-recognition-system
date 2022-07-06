# handwritten-digit-recognition-system
I implemented a CNN to train and test a handwritten digit recognition system using the MNIST dataset. I also read the paper “Backpropagation Applied to Handwritten Zip Code Recognition” by LeCun et al. 1989 for more details, but my architecture does not mirror everything mentioned in the paper. I also carried out a few experiments such as adding different dropout rates, using batch normalization, and using different optimizers in the baseline model. Finally, I discuss the impact of experiments on the learning curves and testing performance.

## Baseline Model
The baseline system uses Glorot initialization, ReLU activations, mini-batch gradient descent with momentum (β = 0.9), early stopping and a cross-entropy loss function. It also uses a learning rate scheduler to adjust the learning rate by 10% every 10 epochs, starting with a learning rate of 0.05.

Learning curves are generated for the validation and training set followed by a discussion whether this baseline system overfits, underfits or reasonably fits the validation data. This baseline system is tested with the testing data and the accuracy is reported along with a confusion matrix. It is available as part of the Jupyter notebook document named: baseline.ipynb.

## Experiments

Incorporated the following changes separately to the baseline model and generated learning curves along with a confusion matrix for the testing set:
- Added dropout using drop rates of 0.25, 0.5 and 0.75
- Incorporated batch normalization before each convolutional hidden layer
- Separately trained using RMSProp, ADAM, and Nesterov optimizers

The above three are available as part of the Jupyter notebook document named baseline-dropout.ipynb, baseline-batch.ipynb, and baseline-optim.ipynb respectively.
