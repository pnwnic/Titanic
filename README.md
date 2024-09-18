This machine learning project aims to utilize logistic regression to solve a binary classification problem. The problem in question is whether or not passengers of the Titanic survived its sinking. To accomplish this, I use two models: Logit from statsmodels and LogisticRegression from scikit-learn.

To begin, I perform EDA on the provided "train" dataset. This step mainly invloved data cleaning, transformations, visualizing distributions and correlation coefficient testing. As a result, I isolated 6 features for prediction.

After isolating and encoding key features, I instantiate a Logit model to validate my feature selection and provide baseline predictions. Model coefficients were plotted to visualize feature influence, followed by significance testing using the P-values of each feature. The findings of this step caused me to drop two additional features.

Next, I split the data into a train and test set, refitting the Logit model to run baseline predictions. The key metric I used for this dataset is F1-score due to the class imbalance of the target variable and the importance of both classes in prediction. The baseline F1-score was 0.73.

The model's precision-recall curve was then visualized to evaluate AUC and optimal F1-score. The result of threshold optimization was an increase in F1 to a score of 0.78. The dicision boundary was then utilized to understand decisions within the model's feature space.

A LogisticRegression model was then instantiated to compare with Logit. LogisticRegression's coefficients were visualized as an overlay for Logit's coefficients to compare differences in understanding of the data. LogisticRegression displayed generally weaker coefficients but similar patterns.

Hyperparameter tuning was applied to LogisticRegression using stratified K-Fold cross-validation in an attempt to boost F1-Score. This resulted in an F1-score identical to the Logit model.

A second attempt to boost F1 was undertaken using synthetic oversampling with SMOTE targeting the minority class. After another round of hyperparameter tuning, the result was a drop in F1-score to 0.75.
A third attempt used random oversampling. This resulted again identically to the max F1-score of 0.78 that had been achieved so far. Shrinkage was then incrementally applied to the oversampled class, but resulted in no change to model performance.

Since there was no increase in performance compared to the Logit model, I used that model to make final predictions. The "test" dataset was then loaded in, cleaned and preprocessed. The result of the model's predictions was an accuracy score of 73.2%.
