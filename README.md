# perth-house-prices
Predicting house prices on the Perth House Prices dataset

# Perth House Prices Prediction

## Table of Contents

1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Univariate Analysis](#univariate-analysis)
   - [Bivariate Analysis](#bivariate-analysis)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
   - [Introducing the SA2 Region Concept](#introducing-the-sa2-region-concept)
   - [Data Transformations](#data-transformations)
   - [Missing Data Imputation](#missing-data-imputation)
   - [Outlier Detection & Removal](#outlier-detection--removal)
5. [Feature Set](#feature-set)
6. [Network Architectures](#network-architectures)
   - [Base Architecture](#base-architecture)
   - [Deep Architecture](#deep-architecture)
   - [Residual Architecture](#residual-architecture)
   - [Wide Architecture](#wide-architecture)
   - [Pyramid Architecture](#pyramid-architecture)
   - [Dense Architecture](#dense-architecture)
7. [Optimisation Strategy](#optimisation-strategy)
8. [Test Results](#test-results)
   - [Reduced Feature Set Training](#reduced-feature-set-training)
   - [Full Feature Set Training](#full-feature-set-training)
   - [Key Findings](#key-findings)
   - [Examining Worst Test Set Predictions](#examining-worst-test-set-predictions)
9. [Recommendations](#recommendations)

---

## Introduction

This assignment explores Perth’s residential property market and predicts house prices using the Perth House Prices dataset. The project aims to build predictive models for Perth house prices using a dataset containing property attributes and their corresponding sale prices. Our approach includes:

- **Exploratory Data Analysis (EDA)**
- **Data cleaning and preprocessing** (handling missing values)
- **Feature importance estimation** via a Random Forest model
- **Development of neural network models**

We compare models built using a reduced set of features (identified as most important by Random Forest) against models employing all available features.

## Data Description

### Data Source & Structure

The dataset consists of **33,656 properties** sold in Perth, each with details such as:

- Address
- Suburb
- Price
- Bedrooms
- Bathrooms
- Garage spaces
- Land and floor areas
- Build year
- Distances to various amenities:
  - Central Business District (CBD)
  - Nearest train station
  - Nearest school
- School’s ranking
- Geospatial coordinates (latitude and longitude)

### Key Columns

- **price (target):** The sale price of the property.
- **bedrooms, bathrooms, garage:** Quantitative room and parking attributes.
- **land_area, floor_area:** Property size indicators.
- **build_year:** Year the property was built.
- **cbd_dist:** Distance to the Perth CBD in metres.
- **nearest_stn_dist:** Distance to the nearest train station.
- **nearest_sch_dist, nearest_sch_rank:** Distance and academic rank of the nearest school.
- **latitude, longitude:** Geographic coordinates of the property.
- **postcode, suburb:** Location identifiers.
- **address:** Unique identifier of the property.

### Data Types & Missing Values

- The dataset comprises **19 columns**, with a mix of numerical (int, float) and categorical (object) fields.
- **Notable missing data:**
  - `garage`: 2,478 missing values.
  - `build_year`: 3,155 missing values.
  - `nearest_sch_rank`: 10,952 missing values.

### Statistical Summary

- **price:** Mean 637,072 AUD, std 355,826, range 51,000 to 2,440,000 AUD.
- **bedrooms:** Typically between 3 to 4 bedrooms.
- **land_area:** Highly skewed, ranging widely up to 999,999 m² (999,999 likely erroneous data), but with a median of 682 m².
- **floor_area:** Median 172 m².
- **cbd_dist:** Median 17,500 m, indicating suburban spread.
- **nearest_sch_dist:** Median 1 km, with the majority of properties relatively close to at least one school.

## Exploratory Data Analysis (EDA)

### Univariate Analysis

- **Price Distribution:** Slightly right-skewed with most properties under 1 million AUD.
- **Rooms and Facilities:** Bedrooms and bathrooms generally cluster around typical family-sized homes (3-4 bedrooms, 1-2 bathrooms). Garage spaces are mostly 2, though outliers exist.
- **Land and Floor Areas:** Land area is highly skewed due to some very large lots. Floor area tends to follow a more normal distribution centered around ~180 m².
- **Build Year:** Properties span a wide historical range with a median year around 1995-2000.
- **Geospatial Features:** Latitude and longitude distributions show concentration within Perth’s urban center.

### Bivariate Analysis

- **Price vs. Bedrooms/Bathrooms:** Increasing bedrooms and bathrooms correlates modestly with higher prices, though with diminishing returns.
- **Price vs. Floor_Area:** A stronger positive correlation (around 0.55) indicates that larger floor areas correspond to higher property values.
- **Price vs. cbd_dist:** Negative correlation (-0.35), suggesting properties closer to the CBD are more expensive.
- **Price vs. nearest_sch_rank:** Negative correlation (-0.46) indicates properties near highly ranked schools (low rank number) tend to command higher prices.
- **Price vs. garage & build_year:** Weaker but positive correlation with garage and mild negative correlations with build_year (indicating newer homes do not necessarily always mean higher prices, possibly due to suburban locations).

#### Strongest Correlations with Price

- `floor_area` (0.55)
- `bathrooms` (0.38) and `bedrooms` (0.25)
- `nearest_sch_rank` (-0.46) and `cbd_dist` (-0.35)

These factors suggest that both internal property features (floor area, number of bathrooms) and external location factors (distance to CBD, quality of nearest school) influence pricing.

We can also visualize where the properties are on a map with prices and land area on the interactive Folium map in the Jupyter Notebook.

**Here are some screenshots at different zoom levels:**

![SA2 Map Western Australia](path/to/western_australia_map.png)
*Figure 1: SA2 Map of Western Australia*

![SA2 Map Metropolitan Perth](path/to/metropolitan_perth_map.png)
*Figure 2: SA2 Map of Metropolitan Perth*

## Data Cleaning and Preprocessing

In this section, we examine ways to augment, clean, and preprocess our dataset to prepare it for model training.

### Introducing the SA2 Region Concept

**What is an SA2 Region?**

- **Definition:** A Statistical Area Level 2 (SA2) is a medium-sized geographical region that generally represents a community with common social and economic characteristics.
- **Size and Coverage:** Each SA2 typically encompasses a population of around 3,000 to 25,000 individuals.
- **Stability and Consistency:** SA2 regions are reviewed and updated less frequently, ensuring stable reference units over time.

**Why Use SA2 Instead of Suburbs or Postcodes Alone?**

1. **Standardisation and Stability:**
   - SA2s are specifically designed for statistical analysis, ensuring stable, well-documented, and methodologically consistent boundaries.
2. **Better Population and Socio-Economic Representation:**
   - SA2 regions capture meaningful patterns, such as local school catchment areas and public amenities.
3. **Dimensionality Reduction and Neighbourhood Context:**
   - Aggregating data to the SA2 level reduces the number of unique categorical units, simplifying the dataset and capturing the essence of a neighbourhood more effectively.

**SA2 Map of Western Australia and Metropolitan Perth**

*Due to the locking of statistical region data behind a paywall, we use an open-source database that contains postcode and corresponding SA data via [Matthew Proctor's Australian Postcodes](https://www.matthewproctor.com/australian_postcodes).*

After merging the SA2 data with our dataset, the number of unique SA2 regions stands at **83**, a significant reduction from suburbs in our dataset (321) while still retaining geographical context.

**Benefits of SA2 Contextual Information:**

- Utilized to impute missing values in the dataset for the `garage` and `build_year` features, instead of using broad imputes like the median count across the whole dataset.

### Data Transformations

- **Log Transformation** on heavily skewed features:
  - `land_area`
  - `nearest_stn_dist`
  - `nearest_sch_dist`
  - `cbd_dist`
- **Date Sold Feature:** Split into two features:
  - `year_sold`
  - `month_sold`

### Missing Data Imputation

- **garage:** ~7% (2,478 values). Imputed with the median garage count at the SA2 level.
  - *Example:* In “Alkimos - Eglinton,” the average garage count was 2, so missing values within that SA2 were set to 2.
- **build_year:** ~9% missing (3,155 values). Imputed with the median build year by SA2.
  - *Example:* Areas like “Baldivis - South” showed an average build year of 2011, so missing values there were filled accordingly.
- **nearest_sch_rank:** ~32.5% missing (10,952 values).
  - Schools eligible to be ranked are ATAR-applicable (Australian Tertiary Admission Rank) schools where students can study WACE (Western Australia Certificate of Education) courses.
  - Missing values imputed with a “bad rank” value of **999** (1 is the best rank) as the nearest school is not eligible to be ATAR-applicable.

### Outlier Detection & Removal

- **Method:** Multivariate Z-score threshold within each SA2 region to detect and remove `price` and `land_area` outliers.
- **Threshold:** 3.5 to flag only extreme outliers, ensuring legitimate high-value properties are retained.
- **Benefit:** Stabilizes model training by reducing the impact of extreme values unlikely to generalize.

## Feature Set

To experiment with the feature set, we identify the most influential predictors using a Random Forest model after data cleaning. Preliminary results indicate:

- **Top Predictors:**
  - `nearest_sch_rank_filled`
  - `floor_area`
  - `cbd_dist_log`
- **Weaker Predictors:**
  - `land_area_log`
  - `sale_year`
  - `build_year`

**Modelling Scenarios:**

1. **Reduced Feature Set Model:** Using only the top 10 predictive features from Random Forest.
2. **Full Feature Set Model:** Using all available features.

*Note:* We use the same train-validation-test split across all model training to prevent any data leakage.

## Network Architectures

### Base Architecture

**BaseNet** implements a straightforward feedforward neural network with three layers:

- Structure: **input → 64 → 32 → 1**
- Features:
  - ReLU activation functions
  - Modest dropout regularization (0.2)

**Purpose:** Serves as an effective baseline for price prediction, allowing for quick training and benchmarking against more complex models.

### Deep Architecture

**DeepNet** extends the basic model with additional capacity and regularization techniques:

- Structure: **input → 256 → 128 → 64 → 1**
- Features:
  - Batch normalization after each hidden layer
  - Dropout implementations

**Purpose:** Captures intricate relationships between housing features while maintaining training stability through batch normalization.

### Residual Architecture

**ResidualNet** introduces skip connections, allowing the network to learn both direct and transformed feature relationships:

- Structure: Consistent width (**128 neurons**) through main processing layers before final prediction

**Purpose:** Preserves important feature information throughout the network, relevant for housing price prediction where some features directly influence price while others require more complex transformation.

### Wide Architecture

**WideNet** emphasizes breadth over depth:

- Structure: Two wide layers of **512 neurons each**
- Features:
  - Extensive feature interaction within each layer
  - Increased dropout rate (0.3) for necessary regularization

**Purpose:** Suitable for capturing complex relationships between multiple housing characteristics, helping prevent overfitting to historical price patterns.

### Pyramid Architecture

**PyramidNet** implements a systematically narrowing structure:

- Structure: **256 → 128 → 64 → 32 → 1**
- Features:
  - LeakyReLU activations

**Purpose:** Creates a controlled feature distillation process, potentially beneficial for gradually extracting relevant pricing factors from raw housing features. Maintains stable gradients throughout training.

### Dense Architecture

**DenseNet** implements connections from each layer to all subsequent layers.

**Purpose:** Allows the network to simultaneously consider features at multiple levels of abstraction, potentially capturing complex interactions between housing characteristics that influence price.

## Optimisation Strategy

The training process employs two optimiser configurations and two learning rate scheduling approaches:

- **Optimisers:**
  - **Adam:** Provides adaptive learning rates for each parameter, valuable for handling features of different scales in housing data.
  - **AdamW:** Adds weight decay regularization, helping prevent overfitting to historical price patterns.

- **Learning Rate Schedulers:**
  - **Plateau:** Adapts learning rates based on validation performance, useful for finding optimal parameters without overfitting.
  - **Cosine:** Provides structured learning rate variation, potentially helping models escape local optima.

*This combination provides flexibility in handling different aspects of the training process.*

## Test Results

### Reduced Feature Set Training

*Details or tables related to training on the reduced feature set would be included here.*

### Full Feature Set Training

*Details or tables related to training on the full feature set would be included here.*

### Key Findings

- **Performance Comparison:**
  - Models trained on the top 10 features selected by the Random Forest model **underperformed** compared to models trained with all features.
  - Suggests intrinsic relations and interactions among the wider set of features.

- **Best Performing Model: `dense_adamw_plateau`**
  - **RMSE:** $136,158.59
  - **R²:** 0.849419 (84.9% of variance explained)
  - **MAE:** $85,274.26

- **Patterns in Model Architecture Performance:**
  - **Dense and Residual architectures** perform best.
  - **Plateau learning rate schedules** generally outperform cosine schedules.

**Interpretation:**

- **R² Value:** Indicates the model explains about 85% of the variance in house prices, capturing most important patterns in the data.
- **Error Metrics:**
  - **RMSE of $136,159:** Shows some significant deviations in individual predictions.
  - **MAE of $85,274:** Indicates that, on average, predictions are off by about $85K.
  - **Difference between RMSE and MAE:** Suggests the presence of outlier predictions where the model makes larger errors.

### Examining Worst Test Set Predictions

1. **Index 15274:**
   - **Property:** Floral farmland
   - **Issue:** Built-up areas (`floor_area`, `bedrooms`, `bathrooms`, `garage`) are for amenities during work hours, not residential.
   - **Impact:** Large land area likely erroneous; models cannot predict well for such outlier cases.

2. **Index 14300:**
   - **Property:** Luxury penthouse in SA2 Fremantle
   - **Issue:** Price is an outlier in the region; dataset lacks enough data points for luxury properties in this SA2.
   - **Impact:** Accurate based on real estate website but not well-predicted by the model.

3. **Index 330:**
   - **Property:** Sale of land with potential for redevelopment
   - **Issue:** Price based on potential appreciation rather than existing residential infrastructure.
   - **Impact:** Model reflects the sale based on land and redevelopment potential, leading to misprediction.

4. **Index 3735:**
   - **Property:** High-priced property in SA2 region
   - **Issue:** Price is on the high side, suggesting the need for log transformation to address skewed distribution.

5. **Index 20914:**
   - **Property:** Trigg – North Beach – Watermans Bay SA2 region
   - **Issue:** High-priced properties indicate possible redevelopment trends; suggests breaking down SA2 into more granular SA1 regions.

## Recommendations

1. **Examine Property Trends in SA2 Regions:**
   - Focus on regions experiencing redevelopment or those too broad to capture specific local contexts.
   - Experiment with breaking down specific SA2 regions into more granular SA1 regions to capture detailed trends.

2. **Adjust Z-score Threshold:**
   - Consider changing the Z-score to exclude more outliers for better model performance.

3. **Remove Outliers:**
   - Specifically target properties that are farmland or have non-residential purposes to improve prediction accuracy.

4. **Log Transformation of Price:**
   - Apply log transformation to better handle the skewed distribution of house prices.

5. **Iterate on Neural Network Architectures:**
   - Further experiment with different architectures, optimisers, and schedulers to enhance model performance.

6. **Explore Non-Neural Network Models:**
   - Investigate other modeling approaches beyond neural networks to potentially capture different patterns in the data.

---

## Links and References

- **SA2 Postcode Data:** [Matthew Proctor's Australian Postcodes](https://www.matthewproctor.com/australian_postcodes)
- **Property Details:**
  - [Property 15274](https://www.realestate.com.au/property/274-pinjar-rd-mariginiup-wa-6078/)
  - [Property 14300](https://www.realestate.com.au/sold/property-house-wa-north+fremantle-124734854)
  - [Property 330](https://www.realestate.com.au/sold/property-house-wa-north+beach-127095818/)
  - [Redeveloped Property](https://www.realestate.com.au/property/1-malcolm-st-north-beach-wa-6020/)

---

*This README was generated to document the Perth House Prices Prediction project, detailing the data exploration, preprocessing, modeling approaches, results, and recommendations for future work.*