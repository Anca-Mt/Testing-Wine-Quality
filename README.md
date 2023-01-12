# Testing Wine Quality

The purpose of this machine learning project is to investigate the potential for using data-driven approaches to predict wine quality. 
The overall objective of this study consists in utilizing different machine learning techniques to predict the quality of wine based on various chemical properties such as acidity, pH, and alcohol content. 
A dataset comprising red or white wine samples and their corresponding quality ratings was exploited for model training and testing. 
The ultimate aim is to establish a reliable model for predicting the quality of new, previously unseen wines. 
In addition to demonstrating the feasibility of using machine learning for wine quality assessment, this project aims to provide valuable insights and practical applications for the wine industry. 


**Methodologies pursued**: 

1 . Use simple machine learning  algorithms such as: Random Forest Classifier, Decision Three Classifier, SVC with different types of kernels

2 . Optimize the model’s performance with the best result by adding Stratified K Fold Model Selection 

3 . Apply Outliers Detection Techniques (Isolation Forest, Elliptic Envelope, Local Outlier Factor or One Class SVM) to determine if the test accuracy would increase


These steps were applied on RED and WHITE datasets (RED = red wine, WHITE = white wine) and the results are presented in the following tables:

** for RED**
![RED](https://user-images.githubusercontent.com/121876169/211009294-5902fc45-7bef-412b-883f-6dda01c7b3c6.jpg)


** for WHITE**
![WHITE](https://user-images.githubusercontent.com/121876169/211013391-054ee276-f741-4811-aa3b-031ac2b6bd4c.jpg)


*I observed that the model for predicting white wine quality did not perform as well as the one for predicting red wine quality. 
To improve this, we analyzed the feature importance and the correlation matrix to identify and eliminate features that were not contributing significantly to the model. 
As a result, we removed the density and total sulfur dioxide features from the dataset, which increased the accuracy from 0.87 to 0.88.*

**WHITE Correlation matrix**
![Correlation_Matrix](https://user-images.githubusercontent.com/121876169/211012471-d8034c57-e539-4401-ac02-7871e0e895bc.png)


**Conclusion**

After analyzing the data, it becomes evident that utilizing Stratified K Fold model selection in conjunction with Random Forest Classifier yields the most favorable results for predicting wine quality in this particular dataset. 
This suggests that a simpler approach, in this case, a combination of these two techniques, may be more effective for achieving the desired outcome.

