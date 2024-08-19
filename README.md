This machine learning project tackles the classic Titanic problem with Scikit-Learn's Logistic Regression model.
I begin by loading in the training dataset and checking for examining null values, employing imputation and transformation techniques to preprocess potential features.
Next, I evaluate the class weights of the target variable and assign them to a dictionary for the model.
Further preprocessing is done using a scaler before instantiating and fitting the base model with class weights and proper solver method assigned.
The model's feature coefficients are plotted using a bar graph to better understand influences.
Cross-validation is performed with a stratified KFold method to discover the best parameters to tune the model.
The tuned model's coefficients are then plotted against the base model for comparison.
To better understand the changes in performance, learning curves for both models are plotted using f1-score.
Data is then split into train/test sets to evaluate the model's confusion matrix and classification report.
At this point, I attempt to improve model performance using synthetic data (SMOTE) to provide more observations for the minority class.
PCA is used to create a scatterplot overlaying the synthetic samples atop the real ones to visualize fit.
Class weights are then confirmed synthetically balanced. A new SMOTE model is fit with new cross-validated parameters and run, resulting in no performance improvements.
The previous model's precision-recall curve is plotted to assist interpretation of isolating the best threshold for the model.
The best threshold is isolated and the new confusion matrix is examined.
The test dataset is then loaded and cleaned for model predictions.
Predictions are made with the best threshold and exported as a CSV.
