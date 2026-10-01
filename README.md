# Spaceship Titanic: Classification and Interpretability

Notebook: [spaceship_titanic.ipynb](spaceship_titanic.ipynb) (text in Portuguese)
Data: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) on Kaggle, 8,693 training passengers.

Predict which passengers were transported to another dimension. The classes are balanced (4,315 vs 4,378), so accuracy, F1 and AUC-ROC are all informative and I selected the final model on AUC-ROC.

## What I did

**Data quality.** Train and test were concatenated so both get identical treatment. `Destination` had 274 nulls; rather than fill with the global mode I imputed hierarchically: same family group first (~120), then the mode within `HomePlanet × Deck` (~149), then within `HomePlanet`, then the global mode. This matters because passengers from Europa go to 55 Cancri e 42.6% of the time and passengers from Mars only 11.2%. Spending columns are zero for more than 60% of passengers, and anyone in CryoSleep spends zero by construction.

**EDA.** CryoSleep is the dominant factor: 82% of sleeping passengers were transported against 33% of those awake. Europa passengers were transported at 66% versus 42% from Earth. Children (0–12) were transported more than average and young adults (18–35) less. Deck B is above 60% and Deck F below 50%; the side of the ship differs by about 6 points.

**Features.** Six groups, each motivated by an EDA finding: `Deck`, `CabinNum`, `Side` split from `Cabin`; `GroupId`, `GroupSize`, `IsAlone` from `PassengerId`; `TotalSpend` and `HasSpent`; `AgeGroup`; `SpendShare` and `GroupTotalSpend`; and `log1p` of every spend column. High-VIF features were kept because the boosted trees handle them with regularisation.

**Models.** Logistic Regression, Decision Tree, Random Forest, XGBoost and LightGBM with 5-fold stratified CV, then GridSearchCV on each. The best model was retrained on all training data and explained with native gain importance and SHAP: beeswarm and bar summaries, the direction of effect for the top five features, and a waterfall plot for one passenger.

## Results

5-fold CV before and after tuning (AUC-ROC):

| Model | Accuracy (pre) | AUC pre-tuning | AUC post-tuning |
|---|---|---|---|
| Logistic Regression | 0.769 | 0.844 | 0.844 |
| Decision Tree | 0.781 | 0.869 | 0.874 |
| Random Forest | 0.798 | 0.890 | 0.892 |
| XGBoost | 0.807 | 0.902 | 0.903 |
| LightGBM | 0.808 | 0.899 | 0.904 |

XGBoost on the hold-out set: accuracy 0.815, F1 0.817, AUC-ROC 0.911. Predicted transport rate on the test set (50.4%) matches the training rate.

After tuning, LightGBM edged XGBoost on CV AUC (0.904 vs 0.903). The gap is well inside fold-to-fold noise and I kept XGBoost, which was ahead before tuning and is the model the interpretability section was built on. By mean absolute SHAP value the five most influential features are `SpendShare`, `log_Spa`, `HasSpent`, `Deck` and `CryoSleep`, so three of the five are engineered spend features, and CryoSleep and Deck line up with the EDA.

## What I would improve

Tuning gained at most 0.005 AUC, so the ceiling here is the features, not the hyperparameters. The group features could go further (for example, the transport rate of other members of the same group, computed out-of-fold). I would also average XGBoost and LightGBM predictions, since their CV scores are indistinguishable.

---

This case is one of seven in my [data science portfolio](https://github.com/juliapmonteirojm-lab/data-science-portfolio).
