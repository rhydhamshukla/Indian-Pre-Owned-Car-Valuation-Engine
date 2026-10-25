# Capstone Major Project Video Presentation Script
Unlox Academy : Machine Learning August 2026 Batch
Student Name: Rhydham Shukla
Batch: ML (Machine Learning) August 2026 Batch
Project Title: Indian Pre Owned Car Valuation Engine and Resale Price Prediction
Target Duration: Around 2 Minutes 30 Seconds to 2 Minutes 45 Seconds
Recording Tool: Windows Snipping Tool (Win + Shift + R) or Game Bar (Win + Alt + R)
Live App URL: http://localhost:8501

Section 1: Self Introduction and Project Overview (0:00 to 0:25)
[On Screen Action]: Share screen showing your Project Title Slide or the top of the Jupyter Notebook.

Hello everyone, and greetings to the Unlox evaluation panel, instructors, and mentors.
My name is Rhydham Shukla, enrolled in the Machine Learning August 2026 Batch at Unlox Academy.
Today, I am presenting my Capstone Major Project: The Indian Pre Owned Car Valuation and Price Prediction Engine.
This is an end to end regression system trained on over 53,000 real world vehicle transactions across major Indian cities, achieving an empirical R squared of 0.9301 and a Mean Absolute Percentage Error of around 11.8 percent.

Section 2: Data Preprocessing and Key Market Insights (0:25 to 1:05)
[On Screen Action]: Scroll through the Data Cleaning and EDA visualizations in the notebook or Tab 3 in the Streamlit app.

In analyzing the Unlox capstone dataset, we first resolved critical real world data issues:
1. Normalizing whitespace in column headers like trailing spaces in Distance.
2. Imputing missing values in engine power, displacement, and seats using brand specific medians.
3. Filtering out target noise below 25,000 rupees.
4. Applying a log transformation on the selling price to stabilize variance and normalize the right skewed price distribution.

During our exploratory analysis, we uncovered two crucial domain insights:
1. Body Type Dynamics: 4 door vehicles in our dataset are mostly executive sedans with a higher median price of 10.4 Lakhs, whereas 5 door models represent compact family hatchbacks with a median of 6.1 Lakhs.
2. Depreciation Driver: Chronological vehicle age is the primary driver of depreciation with a strong negative correlation of 0.51, heavily out weighing secondary factors like ownership transfer count.

Section 3: Feature Synthesis and Model Architecture (1:05 to 1:40)
[On Screen Action]: Show the 5 Fold Cross Validation Benchmark Table in the notebook or report.

To capture multi dimensional depreciation patterns, I engineered domain specific features:
1. Power to displacement ratio to capture engine tuning efficiency.
2. Annual mileage intensity, measuring vehicle stress by dividing kilometers by vehicle age.
3. Quadratic age curves to model accelerated early stage valuation loss.
4. Luxury tier indicators isolating premium brands like BMW, Audi, and Jaguar.

We benchmarked multiple algorithm families under rigid 5 fold cross validation:
1. A baseline Ridge regression achieved an R squared of 0.5912.
2. A Decision Tree reached 0.8415.
3. Tuned gradient boosted models performed significantly better: LightGBM reached 0.9281, XGBoost reached 0.9297, and Random Forest achieved 0.9301.

Combining them into an optimal blended ensemble gave an overall R squared of 0.9301, an RMSE of 223,416 rupees, and an empirical MAPE of 11.83 percent, predicting accurately across both budget cars and luxury SUVs.

Section 4: Live Interactive Web App Demonstration (1:40 to 2:25)
[On Screen Action]: Switch browser tab to the live Streamlit app at http://localhost:8501.

Now let us look at the live production deployment running in our Streamlit web application.

1. Single Vehicle Valuation:
In the first tab, users can select any vehicle. For example, if I select BMW, the model dropdown dynamically filters to verified dataset models: X1, X3, and X5.
Choosing an X1 with 45,000 kilometers, 4 years of age, and manual transmission predicts 16.06 Lakh rupees, which is specifically 1,605,609 rupees on screen.
Upgrading to an X3 adjusts the valuation to 19.52 Lakh rupees.
And upgrading to an X5 predicts 21.15 Lakh rupees.
Notice that every prediction provides a dynamic 5 percent fair value market band and displays consensus estimates across LightGBM, XGBoost, and Random Forest right underneath.

2. Bulk Batch Scoring:
In our second tab, dealerships and platforms can upload the unlabelled test dataset of 5,615 vehicles. Clicking Run Batch Prediction scores the entire dataset in sub seconds, generating the competition submission file: Cap_Submission_Rhydham_Shukla.csv.

Section 5: Conclusion and Wrap Up (2:25 to 2:45)
[On Screen Action]: Switch to your 1 Page Summary PDF or camera.

In conclusion, this project delivers a complete, production grade Machine Learning solution: from data hygiene and statistical domain insights to a verified 0.9301 R squared ensemble and an interactive web deployment.

All deliverables are packaged with:
1. The fully reproducible Colab and Jupyter Notebook
2. The 1 page Academic Summary PDF report
3. The test competition predictions CSV file
4. The complete Streamlit web application source code

Thank you to the Unlox mentorship team for this incredible learning journey. I look forward to the project evaluation and the next phase!
