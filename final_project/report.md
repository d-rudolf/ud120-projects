## Udacity project 'Fraud Identification from Enron Emails'



#### 1. Goal of this project and how machine learning is useful in trying to accomplish it (data exploration, outlier investigation)

The data for this project is partly from the public Enron email dataset (https://www.cs.cmu.edu/~enron/) and partly from financial data from findlaw.com (http://news.findlaw.com/hdocs/docs/enron/enron61702insiderpay.pdf). 
The data can be roughly classified in three classes, which are income and stock data and email statistics. There is also a feature shared receipt with poi, which does fall into any of the three classes. The features are sumerized in Tab. 1.  
<table>  
    <tr>
        <td>income data </td> 
        <td> stock data </td>  
        <td> email statistics </td> 
        <td> misc </td> </tr>
    <tr>
        <td>salary, bonus, deferral payments, deferred income, 
        director fees, expenses, loan advances, long term incentive,
        total payments
        </td> 
        <td> exercised stock options, restricted stock, 
        restricted stock deferred, total stock value 
        </td>
        <td>number of total from/to messages, number of messages
        from/to 
        poi
        </td>
        <td> shared receipt with poi 
     <tr>
</table>
Tab. 1

The goal is to build a model based on the features in Tab. 1 that correctly predicts whether a person is involved in fraud or not (poi = 1 or poi = 0).
To visualize the data I wrote a small web app, which you can run from the flask_app folder:
```
mypath/ud120-projects/final_project/flask_app$ python manage.py runserver
```  
If you define outliers as data points, which are several sigmas away from the mean of a distribution, I found three 
outliers:  'TOTAL', 'LAY KENNETH L', 'SKILLING JEFFREY K'. I did not consider them in my analysis. 


#### 2. Features (create new features, intelligently select features, properly scale features)
I basically selected the financial features: 
'salary', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 'expenses', 'loan_advances', 
'long_term_incentive', 'total_payments'. In addition, I created two new features: 
'from_poi_to_this_person'/'to_messages' and 'from_this_person_to_poi'/'from_messages', i.e. a ratio between  
from_poi and to_poi to the total in and out emails. This reduced the number of features from four to two and creates 
reasonable features. 
Using these features results I can achieve a precision and recall higher than 0.3.       
I selected these features by testing a different subset of features with the tester.py, which creates a  
StratifiedShuffleSplit cross validator and calculates the total accuracy, precision and recall.
Initially, I also tried GridSearchCV, but it gave no agreement with the results from tester.py when defining recall 
or precision as scores. At this point, I also have to make a criticism that originally, in tester.py, precision, recall 
and accuracy are calculated as some kind of global values for all folds of StratifiedShuffleSplit. I think, a better 
approach is to have a distribution of precision, recall and accuracy values where each value comes from a single fold.
Then, its possible to calculate a mean and a standard deviation, which measures the quality of the prediction.        
For scaling, I used the preprocessing.scale, which standardizes the dataset, because algorithms like SVM expect a 
standardized dataset.   


+ What features did you end up using in your POI identifier, and what selection process did you use to pick them?
+ Did you have to do any scaling? Why or why not? 
+ As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset 
-- explain what feature you tried to make, and the rationale behind it. 
+ In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances 
of the features that you use, and if you used an automated feature selection function like SelectKBest, please report 
the feature scores and reasons for your choice of parameter values.

#### 3. Algorithm (pick an algorithm)
+ What algorithm did you end up using? 
+ What other one(s) did you try? How did model performance differ between algorithms?

#### 4. Parameter tuning
+ What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?
+  How did you tune the parameters of your particular algorithm? What parameters did you tune?

#### 5. Valdidation strategy
+ What is validation, and what’s a classic mistake you can make if you do it wrong?
+ How did you validate your analysis?

#### 6. Usage of evaluation metrics
+ Give at least 2 evaluation metrics and your average performance for each of them. 
+  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

