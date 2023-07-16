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
          <ul>
          <li><a href="#1.-Statistical-Inference">1. Statistical Inference</a></li>
          </ul>
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

Dataset is collected and loaded from the directory. After obtaining the dataset, thoroughly examine the data definitions and data types of each feature, which could be categorized as `strings`, `integers`, `floats`, or `objects`. 

To ensure data integrity and prevent any issues with data types or values that fall outside the acceptable range for the trained model, implement <b>data defense mechanisms.</b> This will involve incorporating code to raise a ValueError whenever an unmatched data type or a data value beyond the permissible range is encountered. By doing so, we can maintain the quality and reliability of the data used for training the model.

<p align="center">
<img src="pics/readme-pics/figure-1.png" width=500>
<p>

We will utilize the `sklearn.model_selection.train_test_split` function to divide the dataset into three distinct sets: <b><i>training data, validation data, and testing data.</i></b>

This function will allow us to split the dataset randomly while maintaining the proportions of the data, ensuring that each set is representative of the overall dataset.

<p align="center">
<img src="pics/readme-pics/figure-2.png" width=400>
<p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## EDA and Feature Selection

<p align=center>
<a href="https://github.com/DandiMahendris/regression-model-cac/blob/main/02-eda.ipynb">
  <img src="https://raw.githubusercontent.com/DandiMahendris/regression-model-cac/main/pics/EDA-and-Feature-CAC.jpg" alt="EDA">
</a>
</p>

### 1. Statistical Inference

The given point plot illustrates the relationship between categorical features and the cost (label) data. Although some features appear to have similar means between categories, making it difficult to determine their impact on the label data at a population level, we can conduct statistical inference to gain a more detailed understanding.

To perform statistical inference, we can use techniques like <b><i>Analysis of Variance (ANOVA) or t-tests for categorical variables</i></b>. These methods will help us assess whether the means of the label data are significantly different across the categories of each categorical feature. Here's how we can proceed:

<p align="center">
<img src="pics/readme-pics/image.png" width=500>
<p>

Formulate hypotheses:

- <b>Null hypothesis (H0):</b> There is no significant difference in the means of the label data between the categories of the categorical feature.
- <b>Alternative hypothesis (H1):</b> There is a significant difference in the means of the label data between at least two categories of the categorical feature. <br>

Choose the appropriate statistical test:

- If you have <b>only two categories within each feature</b>, you can perform an independent two-sample t-test.
- If you have <b>more than two categories within each feature</b>, you can perform ANOVA followed by post hoc tests (e.g., Tukey's HSD test) to identify which specific categories differ significantly.
Perform the statistical test and analyze the results:

Calculate the test statistic and p-value.

- If the p-value is below a predefined significance level (e.g., 0.05), we reject the null hypothesis and conclude that there is a significant difference in means between at least two categories.
- If the p-value is not below the significance level, we fail to reject the null hypothesis, indicating that there is no significant difference in means.

Interpret the findings:

- If the null hypothesis is rejected, it suggests that the categorical feature is indeed associated with the label data and may have an impact on the cost.
- If the null hypothesis is not rejected, it implies that the categorical feature may not be significantly related to the label data and may not play a significant role in determining the cost.

<p align="center">
<img src="pics\categoric-selection.webp" width=400>
<p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 2. Parametric Assumption
#### 2.1 Normality
----

`Shapiro-Wilk Test` and `Probability Plot`<br>

- <b>The Shapiro-Wilk </b><br>
&emsp;<b>H<sub>0</sub> (null hypothesis)</b> : the data was drawn from <b>a normal distribution.</b>

The Shapiro-Wilk test is a statistical test that evaluates whether the data is normally distributed. If the p-value resulting from the test is greater than the chosen significance level (commonly set at 0.05), we fail to reject the null hypothesis, indicating that the data is normally distributed. Conversely, if the p-value is less than the significance level, we reject the null hypothesis, suggesting that the data deviates from a normal distribution.

```python
stats.shapiro(model.resid)
```

<b>Shapiro-Wilk Test Result: <br>
ShapiroResult(statistic=0.9924623370170593, pvalue=1.1853023190337898e-40)</b>

However the N > 5000, using probability plot

- <b>Probability Plot</b>

<b>Probability plots, like Q-Q plots (Quantile-Quantile plots),</b> compare the observed data against the expected values from a theoretical normal distribution.

```python
normality_plot, stat = stats.probplot(model.resid, plot= plt, rvalue= True)
```

<p align="center">
<img src="pics/readme-pics/image-1.png" width=400>
<p>

<b>PPCC</b> shown as R2, if R2 is nearly 1 it shown distribution is uniform

<b>PPCC</b> stands for <b><i>Probability Plot Correlation Coefficient.</i></b> PPCC is a measure used to assess the goodness-of-fit of a given probability distribution to a dataset. It quantifies the degree of linear association between the observed data and the theoretical values expected from the specified distribution.

A high PPCC value (close to 1) suggests that the data follows the specified distribution well, while a low PPCC value (close to -1) indicates significant deviations. Other techniques, such as visual inspection or statistical tests like the <b><i>Kolmogorov-Smirnov test</i> or <i>Anderson-Darling test</i></b>

#### 2.2 Homogenity of Variance
----
To evaluate homogeneity of variance, we can use statistical tests like <b><i>Levene's test.</i></b> Levene's test assesses whether the variance of the data significantly differs among the groups defined by the categorical features. 

<b>If the test's p-value is above the significance level, we can assume homogeneity of variance. However, if the p-value is below the significance level, it suggests that the variance is not uniform across the groups.</b>

If <b>assumption violated we can used another non-parametric statistical</b> test such as 
> <mark><b>Welch's ANOVA, Kruskal-Wallis H </b></mark>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 3. One-Way ANOVA Test

<b>Level of significance</b> = α <br>

A one-way ANOVA has the below given null and alternative hypotheses:

- <b>H<sub>0</sub> (null hypothesis)</b>: <br>
&emsp;&emsp; μ1 = μ2 = μ3 = … = μk (It implies that the means of all the population are equal)  <br><br>
- <b>H<sub>1</sub> (alternative hypothesis)</b>:  
&emsp;&emsp; It states that there will be at least one population mean that differs from the rest 

```python
# lst_cate_bool = all more than two-group features
for i,col in enumerate(lst_cate_column):
    model = ols(f'cost ~ C({col})', data=train_set[lst_cate_column]).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    
    model_anova[col] = aov_table['PR(>F)']
    
model_anova_ = (pd.DataFrame(
    data=model_anova.copy(),
    columns=lst_cate_column,
)
                .melt(var_name='columns', value_name='PR(>F)')
                .sort_values(by=['columns'])
                .drop_duplicates()
                .dropna()
)

model_anova_[model_anova_['PR(>F)'] > 0.05]['columns'].values.tolist()
```

<blockquote>
If <mark><b>PR(>F) > 0.05 : Failed to Reject H0</b></mark>, 
that states no significant different mean between independent groups
</blockquote>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 4. Two-Group (T-Test or Welch's Test)

```python
# lst_cate_bool = all two-groups features
for col in lst_cate_bool:
    levene = stats.levene(train_set['cost'][train_set[col]==1],
                                        train_set['cost'][train_set[col]==0])
    print(f'Levene of {col} : \n {levene}')
```

<blockquote>
<b>The <mark>Levene test</mark> examines the H<sub>0</sub> (null hypothesis) that <mark>all input samples originate from populations with equal variances.</mark></b> <br>
</blockquote>

The test results in a non-significant p-value <b>(huge p-value)</b>, indicating a lack of evidence to <b>reject the null hypothesis.</b> <br>
Therefore, we conclude that there is homogeneity of variances among the samples, allowing us to proceed with further analysis.

e.g. <br>
Levene of <b>marital_status:</b> <br>
 &emsp; LeveneResult(statistic=0.34138308811262486, <b>pvalue=0.5590350792841461)</b> <br>
Levene of <b>gender:</b> <br>
&emsp; LeveneResult(statistic=0.740265911515631, <b>pvalue=0.38958058725529066)</b> <br>
Levene of <b>houseowner:</b> <br> 
&emsp; LeveneResult(statistic=3.2592825784464243, <b>pvalue=0.07102729946524858)</b>
  
#### <b>4.1 Independence T-Test</b>
-----
<b>Equal Variance</b> would perform <b>Independence T-Test</b>.<br>
<b>Non-Equal Variance</b> would perform <b>Welch's Test</b>.

        H0 : There's no difference mean between var1 and var2, 
        H1 : There's difference mean between var1 and var2,

><b>H₀ : μ₁ = μ₂</b> <br>
<b>H₁ : μ₁ ≠ μ₂</b>
<br>

        `Independence T-test` used Two-sided alternative with equal_var = True, 
        while `Welch's Test` used Two-sided alternative with equal_var = False

```python
degree = list_0.count() + list_1.count()
    
t_stat, p_value = ttest_ind(list_0, list_1, equal_var=True, alternative="two-sided")
t_crit = scipy.stats.t.ppf(alpha * 0.5, degree)
```
<p align="center">
<img src="pics/readme-pics/image-3.png" width=450)
</p>

All variable on **Equal Variance** is **Failed to Reject H<sub>0</sub>**, then these variable is not statistically significant since mean between group is same <br>

#### 4.2 Welch's Test
-----
```python
degree = list_0.count() + list_1.count()

t_stat, p_value = ttest_ind(list_0, list_1, equal_var=False)
t_crit = scipy.stats.t.ppf(alpha*0.5, degree)
```
<p align="center">
<img src="pics/readme-pics/image-4.png" width=450)
</p>

**Non-Equal variance** group show **Reject H<sub>0</sub>**, then these vairables is statistically significant

#### 4.3 Barplot of Two-Group
------
<p align="center">
<img src="pics/readme-pics/image-5.png" width=450>
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
||Store_cost|total_children|avg_cars_at_home|num_children_at_home | net_weight | units_per_case | coffee_bar | video_store | prepared_food | florist |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| <b>Type</b> | float | float | float | float | float | float | float | float | float | float |
| <b>Data Range </b> | 1700k - 97000k | 0-5 | 0-4 | 0-5 | 3-21 | 1-36 | 0-1 | 0-1 | 0-1 | 0-1 |

<br>

Categorical Data:
|  | promotion_name |sales_country |occupation |avg_yearly_income  | store_type  | store_city  | store_city  | media_type  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| <b>Type</b> | Object | Object | Object | Object | Object | Object | Object | Object | 

