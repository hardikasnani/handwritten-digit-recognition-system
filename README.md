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

## Results

### Baseline System

- In the baseline system, the training and validation loss initially decrease. However, after the early stopping point is reached, the validation loss increases. This hints at the fact that the baseline system is overfitting.
- It's accuracy on test data turned out to be 98.93%.

### Dropout

#### Dropout = 0.25
- The model with dropout set to 0.25 is able to generalize better, which is evident from the fact that the gap between the training loss and validation loss after early stopping point increases to a far lesser intensity than it did for the baseline system.
- It's accuracy on test data turned out to be 99.02%.

#### Dropout = 0.5
- The model with dropout set to 0.50 underfits the data, which is evident from the curves depicting the accuracy and loss of validation data better than the training data. Even though the model generalizes better to the validation data, it does not perform well on the training data.
- It's accuracy on test data turned out to be 98.55%.

#### Dropout=0.75
- The model with dropout set to 0.75 completely underfits the data, which is evident from the curves depicting the accuracy and loss of validation data better than the training data. Even though the model generalizes better to the validation data, it does not perform well on the training data. As compared to the model with dropout set to 0.5, this model underfits at a greater internsity because it has more gap between the training and validation loss.
- It's accuracy on test data turned out to be 95.97%.

The underfitting as seen in the case of dropouts set to 0.50 and 0.75 is due to the bad learning ability of the model using the training data. The best performing model from this group is the one which has dropout set to 0.25. This model is also better than the baseline system because it reduces the amount of overfitting on the data. Based on the test accuracy results, it can be concluded that the model with dropout set to 0.25 has the best test accuracy and its performance is better than all the other models (including baseline system).

### Batch Normalization

- The model with the batch normalization has the validation and training accuracy curves following each other very closely. The validaiton and training accuracy losses are less as compared to the ones in the baseline system. The model with the batch normalization has losses with gaps lesser than what was seen in the baseline system. It's accuracy on test data turned out to be 99.13%.

Based on the results above, it can be said that the model with the batch normalization performs better than the baseline system. When this model is compared with the best model of the previous group i.e. the model with dropout set to 0.25, the accuracy and the loss curves of batch normalized model turns out to be better.

### Optimizer

#### RMSprop Optimizer
- This model was first tested with the parameters that were used in the baseline system. However, it resulted in an extremely poor performance.
- Default paramaters were used for this model and the results achieved were not as good as the ones achieved by the baseline system. It had losses that were greater and accuracy that was lesser than the baseline system.
- It's accuracy on test data turned out to be 98.07%.

#### Adam Optimizer
- This model has the training and validation losses lesser as compared to the baseline system.
- This model and the baseline system have almost similar training and validation accuracies.
- Updating the weights in the network are done more effeiciently when Adam optimier is used. This helps it achieve better performance than the baseline system.
- A version of this model was tried which did not use the learning rate scheduler. However, that model did not perform so well. So, a learning rate scheduler was added and better performance was achieved as compared to the baseline system.
- It's accuracy on test data turned out to be 99.05%.

#### Nestrov Optimizer
- This model has loss and accuracy curves for training and validation that follow looks dimilar to the ones obtained in the baseline system. However, it can be observed that this model has frequent oscillations, but the loss overall is lesser as compared to the baseline system.
- The frequent oscillatory trend in the curve could be due to the nesterov optimizer trying to make big jumps and then correcting itself to move towards the correct gradients.
- It's accuracy on test data turned out to be 98.88%.

The models with Adam optimizer and the Nesterov optimizer have performed fairly well as compared to the model with RMSprop optimizer (with default parameters). Based on the results, the model with Adam optimizer has the best test accuracy and its performance is better than all the other models (including baseline system).

## Best models among the different groups

### Dropout
The model with dropout set to 0.25 outperforms models with dropout set to 0.50 and 0.75.

### Batch Normalization
Since there was only one model, it was the best

### Optimizers
The model with Adam optimizer outperforms all the models with varying other optimizers.

## Best performing model

Considering the three different groups i.e. dropout, batch normalization, and optimizers, it can be safely claimed that the three best models (one from each group) outperformed the baseline system. Also, if a comparison is made between the three best models (one from each group), following can be concluded with:

- The model with Adam optimizer - Best Model
- The model with batch normalization - Second Best Model
- The model with dropout set to 0.25 - Third Best Model
