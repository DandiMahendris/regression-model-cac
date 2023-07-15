# Regression Model - Customer Acquisition Cost - A Feature Selection Approach
----

<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#business-objective">Business Objective</a></li>
    <li><a href="#business-metrics">Business Metrics</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li>
      <a href="#data-workflow">Data Workflow</a>
      <ul>
        <li><a href="#data-preparation">Data Preparation</a></li>
        <li><a href="#eda-and-feature-selection">EDA and Feature Selection</a></li>
        <li><a href="#data-preprocessing-and-feature-engineering">Data Preprocessing and Feature Engineering</a></li>
	<li><a href="#data-modelling">Data Modelling</a></li>      
      </ul>
    </li>
    <li>
      <a href="#prediction-using-api-and-streamlit">Prediction using API and Streamlit</a>
      <ul>
        <li><a href="#how-to-run-by-api">How To Run by API</a></li>
        <li><a href="#data-input">Data Input</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- About the Project -->
## About The Project

<p align=center>
<img src="https://images3.programmersought.com/708/aa/aac300dfe0fc7599cd1dff43f1bd7394.png" width=500>
</p>

### Introduction 

The aim of this project is <b>to explore the application of machine learning models for predicting customer acquisition costs (CAC) and to <mark>investigate the effectiveness of feature selection techniques</mark> in improving the accuracy of these models.</b> <br>
<b>Customer acquisition cost</b> is a crucial metric for businesses, as it directly affects their profitability and marketing strategies. By accurately estimating CAC, companies can <b>optimize their marketing budgets and make informed decisions to maximize return on investment (ROI).</b>

Feature selection plays a vital role in building accurate regression models. It involves identifying the most informative features that have a significant impact on the target variable (CAC). By discarding irrelevant or redundant features, feature selection techniques can enhance the model's performance, reduce overfitting, and improve interpretability.

The research focuses on various feature selection methods, including but not limited to:

- <mark><b>Univariate feature selection:</b></mark> This approach evaluates each feature independently based on statistical measures such as <b><i>chi-square test, ANNOVA test, mutual information, or correlation with the target variable.</i></b>

- <mark><b>Embedded methods:</b></mark> These techniques incorporate feature selection within the model building process itself. For instance, <b><i>Lasso regression</i> performs feature selection and regularization simultaneously.</b>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Business Objective -->
## Business Objective

<b>Customer Acquisition Cost (CAC):</b> This metric represents the average cost a business incurs to acquire a new customer. It includes expenses related to marketing campaigns, advertising, sales efforts, and other customer acquisition activities. 
<p align=center>
<img src="https://stream-blog-v2.imgix.net/blog/wp-content/uploads/80a4679aadc20fd469f22cf3074e79ef/Customer-Acquisition-Cost.png?auto=format&fit=clip&ixlib=react-9.0.3&w=768&q=50&dpr=2" width="350px">
</p>
<br>
The research goal is to investigate the application of machine learning regression models for predicting customer acquisition costs and evaluate the effectiveness of feature selection techniques in improving the accuracy of these models.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Business Metrics -->
## Business Metrics

To evaluate the performance of a machine learning regression model for predicting customer acquisition costs (CAC), we can utilize the following metrics:

1. <b><i>Mean Squared Error (MSE):</i></b> MSE measures the <b>average squared difference between the predicted and actual CAC values.<mark> A lower MSE indicates better model performance,</mark></b> as it indicates that the model's predictions are closer to the actual values.

$$MSE = \frac{1}{n} \sum_{i=1}^{N} (y_i - \hat{y_i})$$

2. <b><i>Coefficient of Determination (R-squared or R2):</i></b> R-squared measures <b>the proportion of the variance in the CAC that can be explained by the regression model.</b> It ranges from 0 to 1, with <b><mark>1 indicating that the model perfectly predicts the CAC and 0 indicating that the model fails to explain any variance.</mark></b> A higher R-squared value signifies a better fit of the regression model to the CAC data.

$$r^2 = 1 - \frac{SSR}{SST}$$

$$SSR = w  \sum_{i=1}^{N} (y_i - \hat{y_i})^2 $$

$$SST = w \sum_{i=1}{N} (y_i - \overline{y})^2 $$

<b><mark>SSR</mark></b> represents the <b>sum of the squared differences between the predicted values (ŷ) and the actual values (y) of the dependent variable in a regression model. </b><br>

<b><mark>SST</b></mark> represents the total sum of squares and quantifies the total variation in the dependent variable. It measures the <b>squared differences between the actual values (y) and the mean of the dependent variable (ȳ)</b>




<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Getting Started -->
## Getting Started

1. Clone the repository
```sh
git clone https://github.com/DandiMahendris/regression-model-cac`
```
2. Install requirement library and package on `requirements.txt`.
3. If you want to create the folder instead
    >   git init <br>
        echo "# MESSAGE" >> README.md <br>
        git add README.md <br>
        git commit -m "first commit"

4. Remote the repository
```sh
git remote add origin git@github.com:DandiMahendris/regression-model-cac.git
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Data Workflow -->
# Data Workflow

## Data Preparation

<p align=center>
<a href="https://github.com/DandiMahendris/regression-model-cac/blob/main/01-pipeline.ipynb">
    <img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/Data-Preparation-CAC.jpg" alt="Preparation" width="300px">
</a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## EDA and Feature Selection

<p align=center>
<a href="https://github.com/DandiMahendris/regression-model-cac/blob/main/02-eda.ipynb">
  <img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/EDA-and-Feature-CAC.jpg" alt="EDA">
</a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Preprocessing and Feature Engineering

<p align=center>
<a href="https://github.com/DandiMahendris/regression-model-cac/blob/main/03-preprocessing.ipynb">
  <img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/Preprocessing-CAC.jpg" alt="Preprocessing" width="350px">
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Modelling

<p align=center>
<a href="https://github.com/DandiMahendris/regression-model-cac/blob/main/04-modelling.ipynb">
  <img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/Modelling-CAC.jpg" alt="Modelling">
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Prediction using API and Streamlit -->
## Prediction using API and Streamlit
<!-- How to Run by API -->
### How To Run by API

1. Open a <b>`Command Prompt`</b> or <b>`PowerShell`</b> terminal and navigate to the folder's directory. Try to test API by following the code:<br>
`$ python .\src\api.py`

<p align="center">
<img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/API-Test.png" alt="api-test-1" width=500>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/API-Test-2.png" alt="api-test-2" width=500>
</p>

2. To try streamlit. Open CMD terminal and type the code: <br>
`$ streamlit run .\src\streamlit.py`

<p align="center">
<img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/Streamlit-Test-1.png" alt="streamlit-test-1" width=500>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/Streamlit-test-2.png" alt="streamlit-test-2" width=350>
</p>


### Data Input

Numerical Data:
|Store_cost|total_children|avg_cars_at_home|num_children_at_home | net_weight | units_per_case | coffee_bar | video_store | prepared_food | florist |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Float | float | float | float | float | float | float | float | float | float |
| Data Range | Data Range | Data Range | Data Range | Data Range | Data Range | Data Range | Data Range | Data Range | Data Range |
| 1700k - 97000k | 0-5 | 0-4 | 0-5 | 3-21 | 1-36 | 0-1 | 0-1 | 0-1 | 0-1 |

<br>

Categorical Data:
| promotion_name |sales_country |occupation |avg_yearly_income  | store_type  | store_city  | store_city  | media_type  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Object | Object | Object | Object | Object | Object | Object | Object | 

