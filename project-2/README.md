# Project 2: Features, Galore: A Linear Regression Analysis on Home Features Predicting Sale Price
---

## Overview 

In this project I looked at which home features will most greatly impact home sale price using Linear Regression. I analyzed an Ames, Iowa Housing Dataset to provide recommendations on which home upgrades are the most important to allocate resources, in order to produce a better return on investment for aspiring house flippers. I extensively cleaned and pre-processed the data prior to running it through different Linear Regression models. The best model was selected. I then used "wild" test data to make sale price predictions to be submitted to Kaggle, a data science and machine learning environment that often hosts data science competitions.   

The data used for this project are from the following datasets: [Training Data](http://localhost:8889/files/project-2/datasets/train.csv?_xsrf=2%7Ca222a359%7C702501f8dda792abc9c0677b0185b861%7C1668018270), [Testing Data](http://localhost:8889/files/project-2/datasets/test.csv?_xsrf=2%7Ca222a359%7C702501f8dda792abc9c0677b0185b861%7C1668018270)

---
## Summary of Findings and Recommendations

The best performing Linear Regression model was the OLS Linear Regression model that included all non multi-collinear features. Regularization techniques, such as Ridge and Lasso, did not out-perform the OLS model in this case. Models that had selected features with a correlation threshold of absolute value 0.3 did not out-perform the OLS Linear Regression model. 

The best model accounted for approximately ~90% of the variation in the target variable, sale price. The best model was able to predict sale price within 27,249 (USD). Analysis of the model residuals showed that the model does not generalize well at either extremes, most notably high home sale price. The ideal prediction range for this model was between 100,000 and 250,000 (USD). The model will need to be adjusted and further tuned in order to better predict outside of this range. 

Insights based on the current model can be drawn to provide recommendations on where to allocate resources in order to yield a favorable return on investment. 

My recommendation is to focus on finding houses with "good bones." Overall quality of features matter more than spending money upgrading every aspect of the house. Focus on upgrading the kitchen with high quality materials, and make sure that care goes into improving the overall quality of the house. Looking for houses with more garage space is likely to produce a high return on investment (ROI), but money does not need to be allocated to improving the garage. Houses with a basement exposed to the outside are likely to yield a higher return on investment, but spending money improving the quality of the basement is also not advised. Having a screened in porch will improve ROI. 

Finally, The neighborhoods I recommend looking for houses are Northridge, Northpark Villa, Briardale, Crawford and Mitchell. These neighborhoods are more middle-priced. They are still likely to improve ROI, but are not top-priced neighborhoods, that may cut too far into the budget for first time home flippers. 

---

## Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|id|int|ames housing data|observation number|
|pid|int|ames housing data|parcel identification number - can be used with city web site for parcel review|
|ms_sub_class|int|ames housing data|Identifies the type of dwelling involved in the sale|
|ms_zoning|object|ames housing data|Identifies the general zoning classification of the sale|
|lot_frontage|float|ames housing data|Linear feet of street connected to property|
|lot_area|int|ames housing data|Lot size in square feet| 
|street|object|ames housing data|Type of road access to property|
|lot_shape|object|ames housing data|General shape of property|
|land_contour|object|ames housing data|Flatness of the property|
|utilities|object|ames housing data|Type of utilities available|
|lot_config|object|hames housing data|Lot configuration|
|land_slope|object|ames housing data|Slope of property|
|neighborhood|object|ames housing data|Physical locations within Ames city limits|
|condition_1|object|ames housing data|Proximity to various conditions|
|condition_2|object|ames housing data|Proximity to various conditions -if more than one is present|
|bldg_type|object|ames housing data|Type of dwelling|
|house_style|object|ames housing data|Style of dwelling|
|overall_qual|int|ames housing data|Rates the overall material and finish of the house|
|overall_cond|int|ames housing data|Rates the overall condition of the house| 
|year_built|int|ames housing data|Original construction date|
|year_remod_add|int|ames housing data|Remodel date-same as construction date if no remodeling or additions|
|roof_style|object|ames housing data|Type of roof|
|roof_matl|object|ames housing data|Roof material|
|exterior_1st|object|ames housing data|Exterior covering on house|
|exterior_2nd|object|ames housing data|Exterior covering on house-if more than one material|
|mas_vnr_type|object|ames housing data|Masonry veneer type| 
|exter_qual|object|ames housing data|Evaluates the quality of the material on the exterior|
|exter_cond|object|ames housing data|Evaluates the present condition of the material on the exterior|
|foundation|object|ames housing data|Type of foundation|
|bsmt_qual|object|ames housing data|Evaluates the height of the basement|
|bsmt_cond|object|ames housing data|Evaluates the general condition of the basement|
|bsmt_exposure|object|ames housing data|Refers to walkout or garden level walls|
|bsmt_fin_type_1|object|ames housing data|Rating of basement finished area|
|bsmt_fin_sf_1|float|ames housing data|Type 1 finished square feet|
|bsmt_fin_type_2|object|ames housing data|Rating of basement finished area-if multiple types|
|bsmt_fin_sf_2|float|ames housing data|Type 2 finished square feet|
|bsmt_unf_sf|float|ames housing data|Unfinished square feet of basement area|
|total_bsmt_sf|float|ames housing data|Total square feet of basement area|
|heating|object|ames housing data|Type of heating|
|heating_qc|object|ames housing data|Heating quality and condition|
|central_air|int|ames housing data|1-has central air, 0-does not have central air|
|electrical|object|ames housing data|Electrical system| 
|1st_flr_sf|int|ames housing data|First Floor square feet|
|2nd_flr_sf|int|ames housing data|Second floor square feet|
|low_qual_fin_sf|int|ames housing data|Low quality finished square feet-all floors|
|gr_liv_area|int|ames housing data|Above grade (ground) living area square feet|
|bsmt_full_bath|float|ames housing data|Basement full bathrooms|
|bsmt_half_bath |float|ames housing data|Basement half bathrooms|
|full_bath|int|ames housing data|Full bathrooms above grade|
|half_bath|int|ames housing data|Half baths above grade|
|bedroom_abv_gr|int|ames housing data|Total rooms above ground|
|kitchen_abv_gr|int|ames housing data|Kitchens above grade|
|kitchen_qual|object|ames housing data|Kitchen quality|
|tot_rms_abv_grd|int|ames housing data|Total rooms above grade (does not include bathrooms)|
|functional|object|ames housing data|Home functionality|
|fireplaces|int|ames housing data|Number of fireplaces|
|garagetype|object|ames housing data|Garage location|
|garage_yr_blt|float|ames housing data|Year garage was built|
|garage_finish|object|ames housing data|Interior finish of the garage|
|garage_cars|float|ames housing data|Size of garage in car capacity|
|garage_qual|object|ames housing data|Garage quality|
|paved_drive|object|ames housing data|Paved driveway||
|pool_area|int|ames housing data|Pool area in square feet|
|misc_feature|object|ames housing data|Miscellaneous feature not covered in other categories|
|misc_val|int|ames housing data|Value of miscellaneous feature|
|mo_sold|int|ames housing data|Month Sold|
|yr_sold|int|ames housing data|Year Sold|
|sale_type|object|ames housing data|Type of sale|
|sale_price|int|ames housing data|Sale price in USD|
|has_pool| int|ames housing data|1-has pool, 0- does not have pool|
|fence|int|ames housing data|fence square footage|
|has_screen_porch|int|ames housing data|1-has screen porch, 0-does not have screen porch|
|has_3season_porch|int|ames housing data|1-has 3 season porch, 0-does not have 3 season porch|
|has_enclosed_porch|int|ames housing data|1-has enclosed porch, 0-does not have enclosed porch|
|has_open_porch|int|ames housing data|1-has open porch, 0-does not have open porch|
|has_wood_deck|int|ames housing data|1-has wood deck, 0-does not have wood deck|

---

