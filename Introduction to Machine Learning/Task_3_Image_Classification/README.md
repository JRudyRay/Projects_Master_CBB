## Image Classification

In task 3 we were able to classify triplets of images with the use of a pretrained model, we used ResNet50, to extract features from the images. 
We further built a neural network for the classification task.

In generate_embeddings() we preprocess the 10000 images and generate embeddings with the resnet50 model.
In get_data() we load the triplet file and map the filenames to the embeddings and generate labels and features for the triplets.
We then defined a model, 'Net'. We tried out different architechtures.
In train_model(), our model gets trained and we use the binary cross entropy loss and adam optimizer to do so. To inspect performance when training we plotted the loss.
In test_model(), the model is tested and performs predictions on the test triplets, which is then saved to results.txt.
