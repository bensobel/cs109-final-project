
# Process Notebook: Premium Illiqudity

## Overview and Motivation

This project has a simple goal: to find correlations for illiquid assets that aren't frequently traded in financial markets with other more liquid, traded assets. 

By using these correlations, synthetic versions of these assets can be constructed to create various types of hedges for people and businesses who have financial exposure to the price of these unsecuritized assets. 

For example, imagine a business that buys raw materials - an ice cream maker that buys milk or a tire company that buys rubber. The prices of commodities like these can be very volatile, and represent a huge fraction of the cost structures of these businesses. A 10% increase in the price of an input could cut the profits of a business like this in half. 

For this reason, many companies are interested in "hedging" their risk to these costs that are outside of their control. These "hedges" should be highly correlated with the underlying asset so they can be bought and sold to move opposite the underlying asset. This minimizes the risk of price movements.

This is a good place to mention the importance of short-selling or "shorting," which is taking a position in the opposite movement of a security. Mechanically this is executed by selling a borrowed share of the security at the date and repurchasing it (also called "closing the position") at a later date. This means that both positive and negative correlations are useful to us because we can take positions in either direction.

While many commodities and other risk factors can be hedged with existing securities, such as commodities futures, some cannot because a liquid spot market does not exist. For this reason, it would be extremely useful for many firms to be able to synthetically "create" such a hedge by purchasing the right combination of assets that closely correlate with the underlying price.






## Related Work

*Anything that inspired you, such as a paper, a web site, or something we discussed in class.*

**Modeling: Inverting the Curse of Dimensionality**

Lectures and labs featuring PCA, LASSO and Ridge Regression enlightened us to the possibility that high dimenssional data might not be a "curse" after all. 

Rather than researching traded securities that might correlate well with a given unsecuritized asset, we went on to collect a vast amount of financial data, which we hoped these algorithms would sift through to retrieve the most pertinent predictors for each unlisted asset.

**Data Transformations**

Course material as well as several research papers caused us to consider the importance of data transformations in adding predictive value to our model. One paper in particular caught our attention in its attempt to predict gold prices from historical data, using transformations rather than raw data (<http://cs229.stanford.edu/proj2013/Potoski-PredictingGoldPrices.pdf>). The use of a moving average, rate of change, and a stochastic oscilator, among other indicators appeared to singificantly improve the model's predictive power.

**Improving the Baseline Model: Leveraging Untraded Predictors**

As we began searching for ways to improve our baseline model, which regressed an unsecuritized asset on a bundle of traded securities, an intriguing question came to mind: Could we employ untradable predictors to improve our model? If we added these untradable predictors as additional independent variables in our regression, the model would lose its interpretability. Rather than representing the amount one must invest in a bundle of tradable securities (weighted by each security's coefficient) in order to replicate the price of an unsecuritized asset, it would signify the amount one would need to invest in untradable predictors *and* tradable securities to track the price of an this asset. 

For example, say we wanted to use data on the supply of polyethylene to improve our model for the price of polypropylene. If we were to add this data as an additional independent variable in our regression, this would mean we would have to "buy" the supply of polyethylene in order to improve our model. This is impossible - one cannot "buy" the supply of polyethylene - it is a statistic, rather than a tradable security. 

Later on, conversations with Professors, Course TFs, Harvard Business School Research Associates and Harvard Innovation Lab Data Scientists revealed innovative ways to leverage untradable predictors to improve our model. We detail our unique modeling approach, which features the use of discretized regressions, in the sections that follow. 

## Initial Questions

*What questions are you trying to answer? How did these questions evolve over the course of the project? What new questions did you consider in the course of your analysis?*

We started and ended this project with one question: can we find bundles of tradable securities that can track the price of unsecuritized assets?

As detailed in the above section, further questions emerged, centered on improving our model: How do we utilize data transformations to improve our model? How can we use untradable predictors to improve our model? How do we prevent and adress overfitting, resulting in very strong in-sample performance but poor out-of-sample results? 

In order to gain insight into these questions, we decided to focus on a particular unsecuritized asset - High Density Polyethylene - and set out to find a bundle of tradable securities that together would track its price.

## Data

*Data on unlisted plastics, traded securities and untradable predictive indicators was downloaded from the data sources listed below. All data was stored on Google Drive and cleaned using Python.*

1. Bloomberg (1990-Present)
    * Polyethylene and Polypropylene consumption, production, net exports and nameplate              capacity
2. Factset (1995-Present)
    * All Futures Contracts - continuous contracts
    * Select Energy Futures Contracts - all maturity dates
3. Thomson Reuters Datastream (1960-Present)
    * Plastics: High Density Polyethylene (HDPE), Low Density Polyethylene (LDPE), 
     Linear Low Density Polyethylene (LLDPE)
    * Bond Indices
    * Currencies
    * Commodity Benchmarks
4. Global Financial Data (1929-Present)
    * Futures Contracts 
    * Stock Indices
    * Commodity Indices
5. Wharton Research Data Services (WRDS) - The Center for Research in Security Prices (CRSP) (1925-Present)
    * Stock Data 
6. Google Trends (2004-Present)
    * Web, Image, News, Shopping and Youtube Search Frequency for polyethylene and its upstream components 
7. The Security and Exchange Commission's (SEC) Electronic Data Gathering, Analysis, and Retrieval system (EDGAR) (1980-Present)
    * Annual Reports of all Companies - *please note this data was scraped*

## Exploratory Data Analysis

*What visualizations did you use to look at your data in different ways? What are the different statistical methods you considered? Justify the decisions you made, and show any major changes to your ideas. How did you reach these conclusions?*




### Preparing the Data for Analysis

**Data Cleaning:** 

We started off by reading in the data files we had collected from the sources detailed in the above section into three Python dataframes: Plastics, Tradable Securities and Untradable Predictors. We then took several steps to clean the dataframes from missing values. 

>First, we truncated the data to span the years 1999-2015. Since our plastics data only goes back to 1986, this was the minimum date we could use as a start date. As a result of the fact that our data is sparser for earlier years, we faced a tradeoff between including more securities and having data series with a longer timespan. After trying several different truncations, we decided that restricting our data to 1999-2015 would be best in terms of insuring that we do not exclude any important predictors, while including a sufficiently long time horizon. 

>Next, in order to make a useful model - one that could be used by businesses in the polyethylene market - we restricted our dataset of securities to currently traded stocks (we defined currently traded stocks as those that have less than 25 missing values in the last year of our dataset). Subsequently, we eliminated all securities with more than 400 missing values (or an average of 25 missing values per year) between 1999 to 2015. We then ran a local linear interpolation function to interpolate missing values using nearby values.

**Data Transformations**

Our next step was to index our dataframes. We indexed prices to make comparisons more consistent so that we are comparing "dollars on dollars." For example, if one stock trades at \$30, and the other trades at \$60, and both increase in price by 10%. In price, they would be \$33 and \$66 respectively, but indexed, both would move from 100 to 110. This means that regressions will compare dollars to dollars, rather than shares to shares, which require coefficients to be converted after-the-fact. (Also, the prices of stocks and most other securities carry little if any information about the company. Companies can issue any quantity of stock at any price, and split or reverse-split the shares outstanding. Shares of stock represent fractions of ownership - the price *combined* with the number of shares provides the total market capitalization of the company, but the price alone is close to meaningless.)

**Train, Validation, Test Split:** 

We then split the data into 3 partitions: a training set, a testing set and a validation set. This split was done on a strict chronological basis, so the first X years were the training set, the next Y the validation set, and the final Z the test set. We chose to employ this chronological split rather than randomly split the data into training, validation and test sets because assuming the data is independently and identically distributed (i.i.d.) would be unwise given that we are dealing with time series data - namely, each point correlates extremely highly with the previous point. 

### Modeling: Stage I:

**Principal Components Analysis (Unsuccessful):** 

Upon recommendation of one of the TFs, we first tried PCA in order to reduce the dimenssionality of our data and to find assets that best account for the variance in the price of plastics. However, due to the high covariance of the data in our dataset, PCA did not yield workable results.

**Lasso:** 

Next, we decided to use regression - the logical choice for a classifier given that our goal of predicting a continuous dependent variable using a bundle of independent variables. Rather than using a simple linear regression, we chose to use Lasso in order to reduce the dimenssionality of our data and return only the most pertinent securities needed to accurately track the price of plastic.

The loss function imposed by Lasso, which reduces many coefficients to zero, was not only important in terms of dimenssionality reduction - greatly reducing the chance of overfitting -  but it was also pivotal in reducing trading costs and increasing the practical feasability of the model. Buying tiny fractional shares of hundreds or thousands of securities may be not only practically infeasible but also expensive due to the transaction costs associated with purchasing each security. 

**Elastic Net:** 

Intuitively speaking, Elastic Net regression is a hybrid of the Lasso and Ridge regressions. Though Lasso and Ridge both handle large numbers of features, Lasso reduces regression coefficients to zero, while Ridge doesn't. With the above discussion regarding transaction costs in mind, it is clear that the output of a Ridge regression would be problematic. However, where Lasso lacks, Ridge shines. While Lasso doesn't deal well with features that covary with each other, Ridge navigates this challenge well - a key strength when dealing with highly covarrying security data. Elastic Net finds a middle ground between these two approaches. Fortunately, the classifier significantly improved results over the LASSO regression.

*Our Baseline model:*
<br><br>
$y_t = \beta_0 + \beta_{1}X_{1t} + \beta_{2}X_{2t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$
<br><br>

show regression equations for Lasso, Ridge and Elastic Net

Graphs... RMSE, R^2

**Moving Averages**

At this point in the project, we realized that the data we had obtained on plastics prices was quite volatile, sometimes moving as much as 40% in a few weeks. This caused us to believe that this data is likely more volatile than the actual price of plastic. For this reason we used a moving average over the data set, using either 180 or 90 day moving averages, fairly standard in technical stock analysis. This helps reduce overfitting from coincidences when other securities happen to jump around at the same time as the plastics data.

**Evaluation Statistics**

To evaluate our models, we considered two standard statistics. The first is Root Mean Squared Error or RMSE. This is functionally the average distance between the prediction and the actual result. It is calculated by taking the error term for each point, squaring it, taking the square root of that, and then calculating the mean of those terms. 

The second is the Pierson $r^2$ coefficient. This term expresses the portion of the 




### Modeling: Stage II

**An Inuitive Model**

While the results of our baseline model were quite encouraging, we saw room for improvement. In particular, we wanted to see if we could use our background financial knowledge to select more relevant securities and avoid overfitting on irrelevant data. This would allow the regressions on their own (without the use of many, many features) to prove themselves, as well as to confirm any intuition we might have. These features included broad stock, bond, and commodity indices, as well as several energy prices (crude oil, natural gas, etc.).

*Our Intermediate model: (Fix this and write it out exactly!)*
<br><br>
$y_t = \beta_0 + \beta_1Oil_t + \beta_2Gas_t + \beta_nSP500_t + \epsilon_t$
<br><br>

where Oil =
Gas = 
SP500 = 
...

Graphs... RMSE, R^2




### Modeling: Stage III

**Technical Indicators:** 

After achieving strong results with the Elastic Net regression, we set out to improve our results through several routes. Our first approach was to collect untraded indicators to improve our model. 

Two of our datasets, CRSP and Global Financial Data, included more granular pricing data, such as "Open Price," "Closing Price," "High Price," and "Low Price" as well as volume - or how many shares of a security are traded in a given day. We calculated several indicators by manipulating these figures. 

One such indicator was a stochastic oscillator, which detects when a stock is overbought or oversold, which we created using a technique adapted from from Potoski, "Predicting Gold Prices," 2013 [http://cs229.stanford.edu/proj2013/Potoski-PredictingGoldPrices.pdf]. We also calculated (Open - Close) and (High - Low), to get at measures of a security's volatility, as well as a rolling 30 day mean, to capture a smoother trend of the stock's trajectory. 

**Language Processing:** 

Next, we used Laguage processing to reduce the dimenssionality of our data by keeping only relavent data series. 

In order to keep only the most relavent stock data, we obtained data via FTP from the SEC's EDGAR - a collection of annual 10-K reports from all publicly-traded companies. We decided to analyze the text of these annual reports in order to determine which companies were most relevant to our focus. 

>We determined a score based on the weighted frequency of the words "High-Density Polyethylene," "Polyethylene," and "Plastics." We then determined a second score based on the occurrence of the words "Methane," and "Ethylene." We calculated this second score because Methane and Ethylene are the upstream components of the plastics we were investigating. Then, the third score we computed was simply a weighted combination of the prior two scores. For each scoring method, we then took the companies with the top 200 scores and took the top 50 correlations from those top 200 scores. We then took a union of the companies present in all three of these top-score lists. From there, we constructed a model using only these top companies from our CRSP data.

We also reduced the dimenssionality of our remaining data - Futures, Currencies, Stock Indecies, and Bond Indecies by keeping only series of percieved relavence to our project. 

>We reduced our futures data to 4 categories of Futures, rather than 10 by eliminating future categories such as Grains and Livestocks - presumably futures that would have low predictive value for our exercise - and keeping Currencies, Energy Interest Rate and Metal Futures.
>We kept only the most broad measure of a currency's performance, and eliminated all measures of a currency against another. Concretely, say the Czech Franque was mentioned 6 times in our dataset: once by intself, i.e. its overall performance, and the other 5 times, it was compared to 5 top currencies, among them the US Dollar and the Euro. We only kept the first mention of the Czech Franque.
> Since the dimenssionality of Stock and Bond Indecies was much lower in comparsisson to these two categories, we left the original datastes as we recieved them.

**Advanced Modeling**

Now that we reduced the dimenssionality of our data through language processing, we attempted to improve our model futher through the use of *untradable* technical indicators. The puzzle was how to integrate them.

If we were to integrate them as additional independent variables in our original regression, our model would no longer achieve its purpose - to find a weighted *traded* basket of securities that would track the price of an unsecuritized asset. Instead, our model would represent a bundle of untraded and traded securities one would have to "invest" in in order to replicate the price of the unlisted asset. Unfortunately it's simply impossible to invest in an untradable asset. 

Say for example, we found that $SupplyP_t$, the supply of polyethylene at time t, was impactful in explaining $P_t$, the price of polyethylene at time t. If we add $SupplyP_t$ to our baseline regression, we see our RMSE decline signifigantly. However, Such a model would not be useful to an individual or business looking to hedge their risk, as they cannot invest in the supply of Polyethylene, $SupplyP_t$!

Therefore, we set out to find a way to somehow leverage these untradable indicators in our model, without actually using them as independent variables in our regression. After talking to professors and practitioners, we came across the phenomenon of discretized regressions.

While we do not use the following discretization in our modeling, we find it is an effective way of explainig the concept. Say you were modeling the function below.


    from IPython.display import Image
    Image(filename='Seasonality.png')
    #reference: http://stats.stackexchange.com/questions/9506/stl-trend-of-time-series-using-r




![png](output_14_0.png)



If you were to draw a line of best fit through the above chart, it seems that the line of best fit is roughly a flat line. Your future pedicitions for years 2006, 2007, etc., would simply be for the price to stay flat. 

However, were you to group the data into yearly increments, fit one regression on each yearly increment, and average the coefficients of your results, your future prediction would be far superior to simply projecting out a horizontal line from 1995 to 2005 and predicting a flat time series for future values.

Mathematically this can be expressed as follows:

$ (1) \quad y_t = \beta_0 + \beta_{1}X_{1t} + \beta_{2}X_{2t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$ 
,$ \quad \quad \quad $ where $\; t \in $ 1995  $\quad \; \ \forall t $

$ (2) \quad y_t = \beta_0 + \beta_{1}X_{1t} + \beta_{2}X_{2t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$ 
,$ \quad \quad \quad $ where $\; t \in $ 1996  $ \quad \; \; \forall t $

$\vdots$

$ (11) \quad y_t = \beta_0 + \beta_{1}X_{1t} + \beta_{2}X_{2t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$ 
,$ \quad \quad \quad $ where $\; t \in $ 2005  $ \quad \forall t $

Given these regressions, average the magnitude of the coefficients, and errors in all regressions to yeild a final regression:

$ (Avg.) \quad y = \beta_0 + \beta_{1}X_{1} + \beta_{2}X_{2} +  ...  + \beta_{n}X_{n} + \epsilon$ 

We take this idea one step further in our analysis. Since our data is not inhrently cyclical in nature as in the above scenario, we could not take advantage of such descretization. However, we propose a novel approach using the power of prediction based on past values.

Say for example that you knew that the price of plastic will increase in the next month. Picture the time series increasing over time. Now fit a linear model to this increasing time series. On the other hand, say that you knew that the price of plastic would decrease in the next month. Picture the time series decreasing over time. Now fit a linear model to this decreasing time series.

It is quite possible that these two scenarios would have different optimal combinations of tradable securities that would best model them. Separating the problem into these two scenarios - where the price of plastic decreases and increases - may thus improve the performance of our model.

But how do we predict whether the price of plastic will increase or decrease in the next month? This is where our untraded predictors come in! We ran a logistic regression **(lets try SVM too if we have time)** to clasify each plastic price at any given time $t$ as a price that will either increase or decrease in the next month based on our untraded *and* traded predictors.

*More concretely the process of our modeling approach is as follows:*

Collect all 30 day windows in which the price of plastic at time $t$ is projected (according to the logistic regression described above) to be less than the price of plastic at time $t+30$. Fit a regresion of the price of plastic on tradable assets to each 30 day window and average the results.

Collect all 30 day windows in which the price of plastic at time $t$ is projected (according to the logistic regression described above) to be greater than the price of plastic at time $t+30$. Fit a regresion of the price of plastic on tradable assets to each 30 day window and average the results.

**We capture the notion above mathematically as follows:**
<br><br>

*Price of plastic increases in time $t+30$ relative to time $t$*

$ (1) \quad y_t = \beta_0 + \beta_{1}X_{1t} + \beta_{2}X_{2t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$ 
,$ \; \; \: \quad \quad \quad \quad \quad \quad $ where $\; \hat{y}_{t+30} > \; y_t $

$ (2) \quad y_{t+1} = \beta_0 + \beta_{1}X_{1t+1} + \beta_{2}X_{2t+1} +  ...  + \beta_{n}X_{nt+1} + \epsilon_{t+1}$ 
,$ \quad \quad \quad $ where $\; \hat{y}_{t+1+30} > \; y_{t+1} $ 

$\vdots$

$ (k) \quad y_{t+k} = \beta_0 + \beta_{1}X_{1t+k} + \beta_{2}X_{2t+k} +  ...  + \beta_{n}X_{nt+k} + \epsilon_{t+k}$ 
,$ \quad \quad \quad $ where $\; \hat{y}_{t+k+30} > \; y_{t+k} $

Given these regressions, we average the magnitude of the coefficients, and errors in all regressions to yeild a final regression:

$ (Avg.) \quad y = \beta_0 + \beta_{1}X_{1} + \beta_{2}X_{2} +  ...  + \beta_{n}X_{n} + \epsilon$ 

<br><br>
*Price of plastic decreases in time $t+30$ relative to time $t$*

$ (1) \quad y_t = \beta_0 + \beta_{1}X_{1t} + \beta_{2}X_{2t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$ 
,$ \; \; \: \quad \quad \quad \quad \quad \quad $ where $\; \hat{y}_{t+30} < \; y_t $

$ (2) \quad y_{t+1} = \beta_0 + \beta_{1}X_{1t+1} + \beta_{2}X_{2t+1} +  ...  + \beta_{n}X_{nt+1} + \epsilon_{t+1}$ 
,$ \quad \quad \quad $ where $\; \hat{y}_{t+1+30} < \; y_{t+1} $ 

$\vdots$

$ (m) \quad y_{t+m} = \beta_0 + \beta_{1}X_{1t+m} + \beta_{2}X_{2t+m} +  ...  + \beta_{n}X_{nt+m} + \epsilon_{t+m}$ 
,$ \quad \quad \quad $ where $\; \hat{y}_{t+m+30} < \; y_{t+m} $

Given these regressions, we average the magnitude of the coefficients, and errors in all regressions to yeild a final regression:

$ (Avg.) \quad y = \beta_0 + \beta_{1}X_{1} + \beta_{2}X_{2} +  ...  + \beta_{n}X_{n} + \epsilon$ 
<br><br>

where $ k+m = n$, the total numbe of samples in our dataset and $\hat{y}_{t}$ denotes the predicted value of $y$ based on the results of the logistic regression below, which incldues both tradable and untradable securities as its independent variables.

$ P(y=1) = \frac{1}{e^-(\beta_0 + \beta_{1}X_{1t} + \beta_{2}X_{2t} +  ...  + \beta_{n}X_{nt} + \epsilon_t)}$



## Final Analysis

"What did you learn about the data? How did you answer the questions? How can you justify your answers?"

### Baseline Model

Our Baseline Model exposed some of the unique challenges of working with financial data. The first encountered in this project is clearly the number of features - over 40,000 were originally encountered, though many were removed from the sample because they did not trade enough within the sample. With a number of features (k) much larger than the number of data points available (n), overfitting is an extreme risk. 

Another factor is missing data points. Many of the securities we obtained data either stopped or started trading within our data set. Without complete data, we need to remove these. Some financial data is excellent, but some is extremely spotty.

But perhaps the most famous problem with finacial data is that it covaries with itself so much. Very simple cyclical macroeconomic factors affect all members of an asset class together: interest rates, risk premia, economic productivity, the price of equities or commodities in general. LASSO regression alone falls victim to this problem, and frequently produced combinations of highly-correlated securities.

Beyond covariance, financial data is victim to the cyclicality of the aforementioned macroeconomic factors, which are, in-fact, cyclical over 5-10 years. With financial data only stretching back often ~10-15 years, this becomes a significant challenge. Imagine trying to produce a model that predicts the weather year-round with only ~18 months of data. 

Finally, the quality of our explanatory data (the plastics data) is highly volatile, likely more so than the actual average realized price of plastic. However, the price of plastic is highly tied to the price of energy, which is itself extremely volatile.

We split our data into testing and training sets, and fit the Elastic Net classifier onto the training set. We were able to fit to our training data with a Root Mean Squared Error (RMSE) of 5.40153910292:
<br><br>
<br><br>
GRAPH HERE
<br><br>
<br><br>
and to our test data with a RMSE of 12.7920612921:
<br><br><br><br>
GRAPH HERE
<br><br><br><br>


### Advanced Model
To construct our advanced model, we used the results of our EDGAR data analysis to select companies that were meaningfully related to polyethylene. <br><br>

This model also seeks to make the leap from explaining prices to predicting prices. This is the difference between finding correlations and causations. Earlier models attempt to make claims on 

## Presentation

Present your final results in a compelling and engaging way using text, visualizations, images, and videos on your project web site.

Leveraging Autoregressive Terms:

$ (1) \quad y_t = \beta_0 + \beta_{1}y_{t-1}
 +  \beta_{2}X_{2t} + \beta_{3}X_{3t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$ 
,$ \quad \quad \quad $ where $\; t \in $ Fall  $ \quad \quad \; \ \forall t $

$ (1) \quad y_t = \beta_0 + \beta_{1}y_{t-1} + \beta_{2}y_{t-2}
 +  \beta_{3}X_{3t} + \beta_{4}X_{4t} +  ...  + \beta_{n}X_{nt} + \epsilon_t$ 
,$ \quad \quad \quad $ where $\; t \in $ Fall  $ \quad \quad \; \ \forall t $
<br><br>


**Heat Map (correlations of independent variables in our bundle of traded securities, with each other):**

**Time series plot of plastics and the prices of securities that are most highly correlated with it:**


    


    


    
