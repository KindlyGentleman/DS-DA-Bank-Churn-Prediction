# Bank Customer Churn Prediction

---

## Business Problem

Customer churn is a major challenge for banks, as it reduces their revenue and market share. It is also costly and difficult to acquire new customers than to retain existing ones. Therefore, banks need to understand the factors that influence customer churn and develop strategies to prevent it.

## Business Objective and Business Metrics

The objective of this project is to create a machine learning model that can accurately predict customer churn and provide insights into the characteristics of customers who are likely to leave the bank. The model will help the bank to target its at-risk customers and offer them personalized incentives or solutions to retain them. The model will also help the bank to improve its products and services based on the feedback and preferences of its customers. The goal is to build a machine learning model that can identify the customers who are likely to leave the bank. This can help the bank to retain its valuable customers and increase its revenue.

## Data Description and Features

The bank has a dataset of 10000 customers with their features and whether they have left the bank or not. The bank wants to use this dataset to build a machine learning model that can predict customer churn based on their features. The bank also wants to identify the most important features that affect customer churn and use them to segment its customers into different groups.

 RowNumber: The record number, which has no effect on the outcome.

- CustomerId: A random identifier, which has no effect on the outcome.

- Surname: The customer’s surname, which has no effect on the outcome.

- CreditScore: The customer’s credit score, which reflects their creditworthiness. A higher credit score means a lower risk of defaulting on loans, and thus a lower likelihood of leaving the bank.

- Geography: The customer’s country of residence (Germany, France, or Spain). Different countries may have different banking preferences and regulations, which can influence the customer’s decision to stay or leave.

- Gender: The customer’s gender (Female or Male). Gender may have some impact on the customer’s banking needs and expectations, which can affect their satisfaction and loyalty.

- Age: The customer’s age. Older customers may have more stable and long-term relationships with the bank, while younger customers may be more willing to switch to other banks for better offers or services.

- Tenure: The number of years that the customer has been with the bank. A longer tenure implies a stronger bond and trust between the customer and the bank, and thus a lower chance of leaving.

- Balance: The amount of money in the customer’s account. A higher balance indicates a higher value and potential for the bank, and also a higher satisfaction and confidence from the customer. A lower balance may signal a lack of interest or engagement with the bank, and a higher possibility of leaving.

- NumOfProducts: The number of products or services that the customer has purchased from the bank. A higher number of products means a more diversified and comprehensive banking experience for the customer, which can increase their retention. A lower number of products may indicate a lack of awareness or satisfaction with the bank’s offerings, which can lead to churn.

- HasCrCard: Whether the customer has a credit card from the bank or not (0 = No, 1 = Yes). Having a credit card can enhance the customer’s convenience and loyalty to the bank, as well as generate more revenue for the bank. Not having a credit card may reduce the customer’s attachment and involvement with the bank.

- IsActiveMember: Whether the customer is an active member of the bank or not (0 = No, 1 = Yes). Active members are more likely to use the bank’s products and services frequently, which can improve their satisfaction and retention. Inactive members may have lower engagement and interest in the bank, which can increase their churn rate.

- EstimatedSalary: The customer’s estimated annual salary. A higher salary means a higher income and spending power for the customer, which can make them more valuable and loyal to the bank. A lower salary may limit the customer’s ability and willingness to use the bank’s products and services, which can make them more likely to leave.

- Exited: Whether the customer has left the bank or not (0 = No, 1 = Yes). This is the target variable that we want to predict using the other features.

## Machine Learning Solution and Metrics

To address the business problem of customer churn, a machine learning solution will be implemented. The solution involves the following steps:

1. Data Preprocessing: The bank's customer dataset will be cleaned and prepared for analysis. This includes handling missing values, encoding categorical variables, and scaling numerical features.

2. Feature Selection: Relevant features that impact customer churn will be identified using exploratory data analysis and feature importance techniques.

3. Model Selection: Various machine learning algorithms will be evaluated, such as Logistic Regression, Random Forest, XGBoost, and LightGBM, to determine the best model for churn prediction.

4. Model Training and Evaluation: The selected model will be trained on the labeled dataset and evaluated using appropriate evaluation metrics such as accuracy, precision, recall, and F1 score.

5. Hyperparameter Tuning: The model's hyperparameters will be tuned using techniques like grid search or random search to optimize its performance.

6. Model Deployment: The trained model will be deployed to a production environment where it can receive new customer data and make real-time predictions.

The performance of the machine learning model will be assessed using the following metrics:

- Accuracy: The overall accuracy of the model in predicting customer churn.
- Precision: The proportion of true positive predictions out of all positive predictions, indicating the model's ability to correctly identify customers who are likely to churn.
- Recall: The proportion of true positive predictions out of all actual positive instances, indicating the model's ability to capture all customers who actually churned.
- F1 Score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.

In conclusion, the machine learning model developed for churn prediction in the bank customer dataset provides promising results. The model achieves an accuracy score of 0.8150 using Logistic Regression, with precision and recall values of 0.8260 and 0.9696 for class 0 (customers who did not exit) and 0.6786 and 0.2394 for class 1 (customers who exited) respectively.

Further analysis of the model's performance reveals that other algorithms such as LightGBM, CatBoost, GBM, RF, XGBoost, KNN, and CART were also implemented and evaluated. Among these, the LightGBM model achieved the highest accuracy score of 0.85333. However, the precision and recall values for class 1 were 0.72414 and 0.49606, respectively. This indicates that while the model performs well in classifying customers who did not exit, there is room for improvement in correctly identifying customers who did exit.

The precision and recall values for class 1 in the XGBoost model, with an accuracy score of 0.84933, were 0.70938 and 0.48819, respectively. This model strikes a good balance between accuracy and correctly identifying customers

## Exploratory Analysis Insights

In conclusion, the churn prediction program for bank customers based on machine learning analysis provides valuable insights into customer behavior and characteristics that influence churn. The program utilizes a dataset of 10,000 customers with 14 variables, allowing for accurate predictions and segmentation of at-risk customers.

The analysis reveals several important findings. Firstly, the average credit score of customers is fair, and lower credit scores are correlated with higher churn rates. This suggests that customers with lower credit scores are more likely to leave the bank.

Secondly, age is a significant factor in customer churn. Older customers, particularly those around 45 years old, are more likely to exit the bank. This insight can help the bank tailor retention strategies to specific age groups.

Thirdly, the number of products used by customers plays a role in churn. Customers using Product 3 and 4 are significantly more likely to leave the bank, indicating potential issues with these products or the bank's ability to meet their needs.

Additionally, customer location is relevant, with customers from Germany more likely to churn compared to other geographical regions. This information can assist the bank in targeting specific regions for retention efforts or exploring reasons behind the higher churn rate in Germany.

Furthermore, owning a credit card or being an active member does not appear to have a significant impact on customer churn. However, female customers are more likely to leave the bank, suggesting a need for gender-specific retention strategies.

In summary, the churn prediction program provides actionable insights for the bank to identify at-risk customers, develop personalized retention strategies, and improve products and services based on customer feedback. By leveraging this machine learning model, the bank can proactively retain valuable customers, reduce churn, and increase overall revenue and market share.

## Strategy Implemented

Based on the information provided, a possible business strategy for the bank could be to focus on retaining customers with lower credit scores, older customers around 45 years old, and customers using Product 3 and 4. The bank could also target specific regions, such as Germany, for retention efforts. Additionally, the bank could consider developing gender-specific retention strategies to retain female customers.

The bank could leverage the insights from the churn prediction program to develop personalized retention strategies for at-risk customers. This could include improving products and services based on customer feedback, offering incentives or promotions to retain customers, and proactively reaching out to at-risk customers to address their concerns.

By implementing these strategies, the bank can proactively retain valuable customers, reduce churn, and increase overall revenue and market share. 

## Streamlit App

![Streamlit](docs/images/Streamlit%20app.png)

## API Testing Using Postman

![Postman](docs/images/Postman%20API%20Testing.png)
Json data
```json
{
   "data":{
      "RowNumber":5,
      "CustomerId":15737888,
      "Surname":"Mitchell",
      "CreditScore":850,
      "Geography":"Spain",
      "Gender":"Female",
      "Age":43,
      "Tenure":2,
      "Balance":125510.82,
      "NumOfProducts":1,
      "HasCrCard":1,
      "IsActiveMember":1,
      "EstimatedSalary":79084.1
   }
}
```