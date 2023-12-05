# AI-Project

Final report 
 
Section 1: Introduction 
 
Team Members: 
Edoardo Brown 281671
Antonio Cagnucci 289871
Omar Regragui 282511
 
Our project focuses on enhancing customer satisfaction prediction for the ThomasTrain Company, merging data science with customer service. We analyzed the "trains_dataset.csv," rich in variables like demographic info, travel details, and service ratings. Our challenge was to predict customer satisfaction indirectly, through data patterns, rather than direct feedback. 
 
We used exploratory data analysis (EDA) to identify trends and relationships. We pre-processed the dataset for the model by addressing missing values, encoding categories, and removing outliers. The goal was to develop machine learning models capable of classifying customers as "satisfied" or "unsatisfied" with high accuracy, ensuring a proper understanding of the underlying factors which affect customer satisfaction. 
 
Our goal was to improve customer retention and our services by employing a blend of data preprocessing, analysis, and advanced machine learning techniques. In the following sections we detail our methods, experimental design and results in a more precise way.  
 
Section 2: Methods 
 
In this section, we outline the methods used for our project. Our approach involved a blend of data handling, visualization, and machine learning techniques.  
 
 
Explanatory Data Analysis (EDA): 
 
Data Handling and Visualization Tools: We started with the basics: utilizing Pandas for managing our dataset. For visualization, we turned to Matplotlib and Seaborn, which were great for mapping basic plots and statistical graphs for example. 
   
Initial Data Inspection: After loading our dataset into a Pandas Data Frame, we used functions like “head()” for a sneak peek into its structure. This step was crucial for setting up our subsequent analysis. 
   
Understanding the Dataset: We dove deeper into the dataset's anatomy by looking at its shape, data types, missing values, and unique values. This comprehensive overview helped us in tailoring our preprocessing strategies. 
 
Visual Exploration: We graphed the distribution of numerical features and the target variable “Satisfied”. Histograms and KDE plots gave us insights into the skewness and spread of our data, allowing us to guide the normalization. 


 <img width="860" alt="histo" src="https://github.com/acagnucci/AI-Project/assets/150381254/022919a6-e982-433a-8fd1-e8dcb850032f">


<img width="378" alt="distribution" src="https://github.com/acagnucci/AI-Project/assets/150381254/591397d6-5808-4f35-b114-1d03cb8c764a">



Data Transformation: The categorical 'Satisfied' column was encoded into numerical values, a necessary step for the classification models we intended to use. 
 
Correlation Analysis: We then plotted a heatmap, where we could see how different features related to customer satisfaction. It helped us pinpoint which variables might be key players. 


<img width="829" alt="m1" src="https://github.com/acagnucci/AI-Project/assets/150381254/eac4e97c-7e3e-4c0b-9024-80a069a751b2">

 
 
Data Preprocessing and Pipeline Construction: 
 
Feature Selection and Splitting: Post-EDA, we separated our target variable “Satisfied” and identified key numerical and categorical columns for modeling. 
 
Pipeline Setup: We built a machine learning pipeline using Scikit-learn's “Pipeline” and “ColumnTransformer”. This helped us neatly bundle our preprocessing steps, handling numerical and categorical data separately but in parallel. 
 
For numerical data, we used “SimpleImputer” for missing values and “StandardScaler” for normalization. 
For categorical data, “SimpleImputer” and “OneHotEncoder” took care of missing values and transformation, respectively. 
 
Training and Test Split: We split our dataset into training and test sets (80-20 split), and further partitioned the training data to create a validation set. This setup was essential for unbiased model evaluation. 
 
 
Further EDA and Machine Learning Warnings: 
Deeper Dive into Data: We revisited our numerical data, plotting histograms to catch any nuances we might have missed initially. The correlation matrix also got a second look, helping us refine our feature selection. 
   
Handling Machine Learning Warnings: We encountered convergence warnings with logistic regression, which we addressed by adjusting the number of iterations and ensuring proper data scaling. 


 <img width="1185" alt="h2" src="https://github.com/acagnucci/AI-Project/assets/150381254/08d9de25-48f6-4a31-9a7d-cf9455c5df8a">


<img width="765" alt="m2" src="https://github.com/acagnucci/AI-Project/assets/150381254/04cafc52-5c13-4744-a913-ab1cf6ed2c4b">


 
Model Training and Hyperparameter Tuning: 
 
Model Selection: We chose Logistic Regression, Decision Tree, and Random Forest for our classification task. Each brought a unique perspective to the table. 
 
Training and Initial Evaluation: We trained each model on our dataset, evaluating their performance using a validation set. This was our first checkpoint to see how our models fared with the default settings. 
 
Tuning Hyperparameters: Using “RandomizedSearchCV”, we fine-tuned our models. For example, we adjusted the Decision Tree's depth, sample splits, leaf samples, and criterion based on accuracy. 
 
Learning Curves and Performance Evaluation: We plotted learning curves to understand our models' behavior with increasing data. The final performance was evaluated on the test set using metrics like accuracy, precision, recall, F1-score, and ROC-AUC score. 
 
Section 3: Experimental Design 
 
Introduction to Experimental Approach: 
Our experimental design focused on accurately predicting customer satisfaction and identifying the most impactful features. This approach encompassed comprehensive analysis, model selection, and evaluation. 
 
Choice of Models and Baseline Establishment: 
Logistic Regression: Selected for its computational efficiency and interpretability, serving as a foundational benchmark. 
Decision Trees: Chosen for their ability to model non-linear relationships and interpretability, without the need for feature scaling. 
Random Forest: An ensemble of Decision Trees, aimed to enhance performance and stability, reducing overfitting risks and better handling varied features and interactions. 
 
Detailed Evaluation Metrics: 
Accuracy: Assessed the overall performance of the models. 
Precision: Important for minimizing false positives in customer satisfaction prediction. 
Recall: Important for correctly identifying all instances of dissatisfaction. 
F1-Score: Provided a balanced metric for precision and recall, especially vital in an imbalanced dataset. 
ROC-AUC Score: Measured the models' ability to distinguish between satisfied and unsatisfied customers. 
 
These measurements were all very important for a complete review of the models. 
 
Hyperparameter Tuning and Validation: 
We employed a randomized search for hyperparameter tuning, offering an efficient alternative to grid search. 
 
Learning Curve Analysis: 
Learning curves helped us properly understand our model’s performance, identifying issues like overfitting and underfitting, and helping us make our decisions on how to tune our model. 
 
Final Model Selection Based on Validation Performance: 
We analyzed the validation performance of each model using the metrics mentioned earlier. The best-performing models were then re-trained on the full training dataset, including the validation data, to fully harness their predictive power. 
 
Test Set Evaluation for Generalization: 
The final evaluation phase involved deploying our tuned models on the test set. This step was crucial in assessing their generalization capabilities and ensure it’s intended function. 
 
Conclusive Remarks: 
Our experimental design was meticulously crafted to ensure a balance between predictive power and interpretability. By selecting appropriate models, evaluation metrics, and methodologies, we aimed to create a robust and comprehensive approach, tailored to the project's needs in predicting customer satisfaction. 
 
 
# Section 4: Results 
 
Comparative Model Analysis: 
 
Upon the comprehensive evaluation of the models, the Random Forest and Decision Tree classifiers demonstrated considerable predictive capabilities in forecasting customer satisfaction for the ThomasTrain company. The Random Forest model, with its ensemble approach, achieved a good accuracy of 85%, showcasing its robustness through high precision (0.97) for the positive class (satisfied customers). However, it faltered slightly in recall (0.67), suggesting a tendency to overlook certain satisfied customers. The ROC AUC score of 0.942 indicated a high ability to differentiate. 
 
The Decision Tree model, on the other hand, presented a superior balance across all metrics, achieving an impressive accuracy of 95%. With precision and recall both above 0.93 and an F1-Score of 0.94, this model demonstrated an exceptional ability to identify satisfied customers without significant compromise. Its ROC AUC score of 0.981 suggested excellent classification capabilities. 
 
Learning Curve Interpretations: 
The learning curves provided further insight into model performance. For the Random Forest, the initial disparity between training and validation scores indicated overfitting; however, additional training data seemed to lower this gap, improving model generalization. The Decision Tree's learning curves showed a strong start with less variance between training and validation scores, indicating a stable and consistent performance that was maintained as more data was introduced. 



 <img width="399" alt="1" src="https://github.com/acagnucci/AI-Project/assets/150381254/d210e915-0192-4d30-aa9b-283a8b3b1f44">
<img width="393" alt="2" src="https://github.com/acagnucci/AI-Project/assets/150381254/67466122-5d02-4b95-80cc-5647300356e7">



Justification for Model Selection: 
Considering all metrics and learning curve analyses, the Decision Tree was selected as the preferred model. Its exemplary balance of precision and recall, coupled with the highest F1-Score and ROC AUC score, indicated a model well-fitted to the data without overcomplicating the underlying structure. Furthermore, its learning curve suggested that the model was learning efficiently and was less likely to benefit from additional data. 
 
Interpretation of Model Outcomes: 
The performance metrics indicate that while the Random Forest model was more cautious in predicting satisfaction, leading to high precision but lower recall, the Decision Tree model managed to maintain high standards across all metrics. This suggests that the Decision Tree model was able to capture the complexity of customer satisfaction without overfitting or underfitting. 
 

 <img width="680" alt="Screenshot 2023-12-05 at 20 34 36" src="https://github.com/acagnucci/AI-Project/assets/150381254/360b7577-1657-4a3f-94fb-acd42639e94e">

 
 
Conclusion: 
In the final analysis, while both models showed strengths, the Decision Tree model's exceptional performance across precision, recall, F1-Score, and ROC AUC score, along with its learning curve profile, affirmed its selection as the optimal model for this project. It demonstrated not only high accuracy but also the capability to generalize well, ensuring reliability in predicting customer satisfaction for the ThomasTrain company. 


# Section 5: Conclusions 
 
Synthesis of Project Insights: 
 
We concluded that the Decision Tree model is the best tool for predicting customer satisfaction for the ThomasTrain company. This model not only achieved high accuracy but also provided proper interpretability that can be used from a business point of view. It has proven to be adept at navigating the complexity of customer data, offering a balance between the precision of predictions and the ability to capture most satisfied customers. The essential take-away from our work is that customer satisfaction can indeed be predicted with a significant degree of accuracy using machine learning models. Moreover, the project underscores the importance of choosing models that not only perform well statistically but also align with business goals and practical usability. 
 
Unanswered Questions and Future Directions: 
 
While the Decision Tree model worked well for us, it makes us wonder if a more complex model could find deeper insights without making it hard to understand. Also, we didn't fully check how customer satisfaction changes over time and if models could learn and predict these changes. We could've also used unsupervised learning to find hidden patterns in the customer base. 
 
For future work, we should think about using models that can handle data over time, like recurrent neural networks, or try unsupervised learning for customer segmentation. Also, we should dig deeper into how features are created to find more relationships in the data. Adding external information, like economic indicators or trends in transportation, might make the model better. We should keep checking how well the model works as customer behaviors and expectations change. 
 


 
 
 
