# AI-Project
This is the machine learning project 281671


# Section 1: Introduction 
 
Team Members: 
Edoardo Brown 
Antonio Cagnucci 
Omar Regragui 
 
Our project focuses on enhancing customer satisfaction prediction for ThomasTrain Company, merging data science with customer service. We analyzed the "trains_dataset.csv," rich in variables like demographic info, travel details, and service ratings. Our challenge was to predict customer satisfaction indirectly, through data patterns, rather than direct feedback. 
 
The project involved rigorous exploratory data analysis (EDA) to identify trends and relationships. We prepared the dataset for predictive modeling, addressing missing values, encoding categories, and outlier removal. The goal was to develop machine learning models capable of classifying customers as "satisfied" or "unsatisfied" with high accuracy, ensuring a deep understanding of influential factors in customer experiences. 
 
This undertaking aims to guide ThomasTrain in enhancing its services and customer retention, employing a blend of data preprocessing, analysis, and advanced machine learning techniques. The subsequent sections detail our methodologies, experimental designs, and the insights we gained, contributing significantly to customer satisfaction prediction in the train industry.  
 
# Section 2: Methods 
 
In this section, we outline the methodologies we adopted for our project. Our approach was multi-faceted, involving a blend of data handling, visualization, and machine learning techniques. 
 
Explanatory Data Analysis (EDA): 
Data Handling and Visualization Tools: We started with the basics, utilizing Pandas for managing our dataset. For visualization, we turned to Matplotlib and Seaborn, which were great for everything from basic plots to more complex statistical graphs. 
   
Initial Data Inspection: After loading our dataset into a Pandas Data Frame, we used functions like “head()” for a sneak peek into its structure. This step was crucial for setting up our subsequent analysis. 
   
Understanding the Dataset: We dove deeper into the dataset's anatomy by looking at its shape, data types, missing values, and unique values. This comprehensive overview was vital for tailoring our preprocessing strategies. 
 
Visual Exploration: We graphed the distribution of numerical features and the target variable “Satisfied”. Histograms and KDE plots gave us insights into the skewness and spread of our data, guiding our normalization efforts. 

![.](images/histo.png)
![.](images/distribution.png)
 
Data Transformation: The categorical 'Satisfied' column was encoded into numerical values, a necessary step for the classification models we intended to use. 
 
Correlation Analysis: This was a game-changer. By plotting a heatmap, we could see how different features related to customer satisfaction. It helped us pinpoint which variables might be key players. 
![.](images/m1.png)
 
Data Preprocessing and Pipeline Construction: 
Feature Selection and Splitting: Post-EDA, we separated our target variable “Satisfied” and identified key numerical and categorical columns for modeling. 
 
Pipeline Setup: We built a machine learning pipeline using Scikit-learn's “Pipeline” and “ColumnTransformer”. This helped us neatly bundle our preprocessing steps, handling numerical and categorical data separately but in parallel. 
 
  For numerical data, we used “SimpleImputer” for missing values and “StandardScaler” for normalization. 
  For categorical data, “SimpleImputer” and “OneHotEncoder” took care of missing values and transformation, respectively. 
 
Training and Test Split: We split our dataset into training and test sets (80-20 split), and further partitioned the training data to create a validation set. This setup was essential for unbiased model evaluation. 
 
Further EDA and Machine Learning Warnings: 
Deeper Dive into Data: We revisited our numerical data, plotting histograms to catch any nuances we might have missed initially. The correlation matrix also got a second look, helping us refine our feature selection. 
![.](images/h2.png)
![.](images/m2.png)
   
Handling Machine Learning Warnings: We encountered convergence warnings with logistic regression, which we addressed by adjusting the number of iterations and ensuring proper data scaling. 
 
Model Training and Hyperparameter Tuning: 
Model Selection: We chose Logistic Regression, Decision Tree, and Random Forest for our classification task. Each brought a unique perspective to the table. 
 
Training and Initial Evaluation: We trained each model on our dataset, gauging their performance using a validation set. This was our first checkpoint to see how our models fared with the default settings. 
 
Tuning Hyperparameters: Using “RandomizedSearchCV”, we fine-tuned our models. For example, we tweaked the Decision Tree's depth, sample splits, leaf samples, and criterion based on accuracy. 
 
Learning Curves and Performance Evaluation: We plotted learning curves to understand our models' behavior with increasing data. The final performance was evaluated on the test set using metrics like accuracy, precision, recall, F1-score, and ROC-AUC score. 
 
# Section 3: Experimental Design 
 
Introduction to Experimental Approach: 
Our experimental design focused on accurately predicting customer satisfaction and identifying the most impactful features. This approach encompassed comprehensive analysis, model selection, and evaluation. 
 
Choice of Models and Baseline Establishment: 
Logistic Regression: Selected for its computational efficiency and interpretability, serving as a foundational benchmark. 
Decision Trees: Chosen for their ability to model non-linear relationships and interpretability, without the need for feature scaling. 
Random Forest: An ensemble of Decision Trees, aimed to enhance performance and stability, reducing overfitting risks and better handling varied features and interactions. 
 
Detailed Evaluation Metrics: 
Accuracy: Assessed the overall performance of the models. 
Precision: Important for minimizing false positives in customer satisfaction prediction. 
Recall: Crucial for correctly identifying all instances of dissatisfaction. 
F1-Score: Provided a balanced metric for precision and recall, especially vital in an imbalanced dataset. 
ROC-AUC Score: Measured the models' ability to distinguish between satisfied and unsatisfied customers. 
 
These metrics were integral to a holistic assessment of the models. 
 
Hyperparameter Tuning and Validation: 
We employed a randomized search for hyperparameter tuning, offering an efficient alternative to the more exhaustive grid search. 
 
Learning Curve Analysis: 
Learning curves helped us understand model performance dynamics, identifying issues of high variance (overfitting) or high bias (underfitting), and guiding our decisions on data augmentation or model complexity adjustments. 
 
Final Model Selection Based on Validation Performance: 
 
We analyzed the validation performance of each model using the metrics mentioned earlier. The best-performing models were then re-trained on the full training dataset, including the validation data, to fully harness their predictive power. 
 
Test Set Evaluation for Generalization: 
 
The final evaluation phase involved deploying our tuned models on the test set. This step was crucial in assessing their generalization capabilities and real-world applicability. 
 
Conclusive Remarks: 
 
Our experimental design was meticulously crafted to ensure a balance between predictive power and interpretability. By selecting appropriate models, evaluation metrics, and methodologies, we aimed to create a robust and comprehensive approach, tailored to the project's needs in predicting customer satisfaction. 
 
 
# Section 4: Results 
 
Comparative Model Analysis: 
 
Upon the comprehensive evaluation of the models, the Random Forest and Decision Tree classifiers demonstrated considerable predictive capabilities in forecasting customer satisfaction for the ThomasTrain company. The Random Forest model, with its ensemble approach, achieved a commendable accuracy of 85%, showcasing its robustness through high precision (0.97) for the positive class (satisfied customers). However, it faltered slightly in recall (0.67), suggesting a tendency to overlook certain satisfied customers. The ROC AUC score of 0.942 indicated strong discriminative power. 
 
The Decision Tree model, on the other hand, presented a superior balance across all metrics, achieving an impressive accuracy of 95%. With precision and recall both above 0.93 and an F1-Score of 0.94, this model demonstrated an exceptional ability to identify satisfied customers without significant compromise. Its ROC AUC score of 0.981 suggested excellent classification capabilities. 
 
Learning Curve Interpretations: 
The learning curves provided further insight into model performance. For the Random Forest, the initial disparity between training and validation scores indicated overfitting; however, additional training data seemed to mitigate this gap, improving model generalization. The Decision Tree's learning curves showed a strong start with less variance between training and validation scores, indicating a stable and consistent performance that was maintained as more data was introduced. 

![Random forests learning curve:](images/1.png)
![Decision trees learning curve:](images/2.png)
 
   
 
 
Justification for Model Selection: 
Considering all metrics and learning curve analyses, the Decision Tree was selected as the preferred model. Its exemplary balance of precision and recall, coupled with the highest F1-Score and ROC AUC score, indicated a model well-fitted to the data without overcomplicating the underlying structure. Furthermore, its learning curve suggested that the model was learning efficiently and was less likely to benefit from additional data, which is an important consideration in a production environment where the economy of data can be a constraint. 
 
Interpretation of Model Outcomes: 
The performance metrics indicate that while the Random Forest model was more cautious in predicting satisfaction, leading to high precision but lower recall, the Decision Tree model managed to maintain high standards across all metrics. This suggests that the Decision Tree model was able to capture the complexity of customer satisfaction without being overly stringent or too lenient in its predictions. 
 
![.](images/Screenshot%202023-12-05%20at%2020.34.36.png)
 
 
Conclusion: 
In the final analysis, while both models showed strengths, the Decision Tree model's exceptional performance across precision, recall, F1-Score, and ROC AUC score, along with its learning curve profile, affirmed its selection as the optimal model for this project. It demonstrated not only high accuracy but also the capability to generalize well, ensuring reliability in predicting customer satisfaction for the ThomasTrain company. 
 
 
# Section 5: Conclusions 
 
Synthesis of Project Insights: 
 
We concluded that the Decision Tree model is the best tool for predicting customer satisfaction for the ThomasTrain company. This model not only achieved high accuracy but also provided proper interpretability that can be used from a business point of view. It has proven to be adept at navigating the complexity of customer data, offering a balance between the precision of predictions and the ability to capture most satisfied customers. The essential take-away from our work is that customer satisfaction can indeed be predicted with a significant degree of accuracy using machine learning models. Moreover, the project underscores the importance of choosing models that not only perform well statistically but also align with business goals and practical usability. 
 
Unanswered Questions and Future Directions: 
 
 
While the Decision Tree model worked well for us, it makes us wonder if a more complex model could find deeper insights without making it hard to understand. Also, we didn't fully check how customer satisfaction changes over time and if models could learn and predict these changes. We could've also used unsupervised learning to find hidden patterns in the customer base. 
 
For future work, we should think about using models that can handle data over time, like recurrent neural networks, or try unsupervised learning for customer segmentation. Also, we should dig deeper into how features are created to find more relationships in the data. Adding external information, like economic indicators or trends in transportation, might make the model better. We should keep checking how well the model works as customer behaviors and expectations change. 
 
 
 
 
 
