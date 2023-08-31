## Transfer Learning 

In task 4 we applied transfer learning to predict the HOMO-LUMO gap from molecule descriptions.
In load_data, all the datasets are loaded from the zip files.
We then defined our model architechture, which is a feed forward neural network with three fully connected layers
including ReLU activations and dropout layers.
In train_model we then train the model with the training data.
In validate_model we validate the model on the validation data.
In make_feature_extractor we train the extractor on the pretraining data which allows us to extract features from the training and test data.
We then include make_pretraining_class, which integrates the feature extraction into the pipeline.
In get_regression_model we then define the regression model we used. We tried many different models and got the best result using the RidgeCV model.
In the main part we then execute all the functions, predicting the HOMO-LUMO gap and saving the results to a .csv file.
