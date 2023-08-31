## Feature Transformation and Ridge Regression

In the transformation function transform_data(X) we first defined the individual transformations that will be applied and stored them in a variable to easily call upon in the loop where we perform the transformations.
We thereby transform the original 5 features into 21 features and thereby obtain a transformed feature matrix with dimensions (700,21).

For the rest of the functions we proceeded the same as in task 1a, but we tested it on more lambda values to be able to select an optimal lambda (best_lambda), 
to arrive at our solution. the best lambda is chosen based on the one that gives rise to the lowest average root mean squared error.
