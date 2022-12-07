###############################
## CMSE 830 Final Project ##
## Wenting Liu               ##
## Dec. 2022                 ##
###############################

# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix


# Title and author name:
st.markdown("## Identification of feature candidates associated with preterm birth through analysis of cord blood data (ML Part)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Wenting Liu")
with col2:
    st.markdown("### Dec. 2022")
st.text("")

## Workflow:
st.markdown("### 0. The Pipeline for Building Machine Learning Classification Models")
wf = st.button('Pipeline')
if wf:
    st.markdown("""![Pipeline](https://raw.githubusercontent.com/LoWeT0619/CMSE_830_Final_Proj/main/Pipeline-v2.png)""")

# 1. Loading Dataset and Data Overview:
st.markdown("### 1. Loading Dataset and Data Overview")

## 1.1 Loading Dataset:
st.markdown("#### 1.1 Loading Dataset")
col1, col2 = st.columns(2)
with col1:
    bg = st.button('Background')
if bg:
    st.markdown("""
    - [**Preterm birth**](https://www.cdc.gov/reproductivehealth/maternalinfanthealth/pretermbirth.htm#:~:text=Preterm%20birth%20is%20when%20a,2019%20to%2010.1%25%20in%202020.) is when a baby is born too early, **before 37 weeks of pregnancy** have been completed. It is known to be associated with chronic disease risk in adulthood. Gestational age (GA) is the most important prognostic factor for preterm infants. In this project, we would like to elucidate which features (include clinical features and cell type features) are related to preterm birth (which is when GA < 37 weeks in this project).

    - With those identified feature candidates, we could then conduct to Epigenome-wide association studies (EWAS). By associating the identified feature candidates with their related DNA information, we could then figure out which gene changed may lead to preterm birth. Then in the future, maybe researchers will find out some targeted medications that can help solve the preterm birth issue or do some risks prediction to make it controllable.
    """)

with col2:
    ques = st.button('The question')
if ques:
    st.latex(r'''
        y = \alpha + \beta_{1} x_{1} + \beta_{2} x_{2} + ... + \beta_{n} x_{n} + \epsilon
    ''')

    st.latex(r'''
        preterm birth = \alpha + \beta_{1} feature_{1} + \beta_{2} feature_{2} + ... + \beta_{n} feature_{n} + \epsilon
    ''')
st.text("")

## 1.2 Data Overview:
st.markdown("#### 1.2 Data Overview")
col1, col2 = st.columns(2)
with col1:
    data_intro = st.button('Data Introduction')
if data_intro:
    st.markdown("""
    - The dataset comes from the [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7873311/) called *Identification of epigenetic memory candidates associated with gestational age at birth through analysis of methylome and transcriptional data* by Kashima et al. It was published in *Scientific Reports* (The 2021-2022 Journal's Impact is 4.379) in September 2021.

    - The paper above shared its related data on [NCBI](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110828).

    - The data contains:
    
        - Phenotype dataset (110, 23): 
        
            - Rows: 110 cord blood samples ID
            
            - Columns: 23 features:
            
                - 16 clinical features (6 continuous + 10 discrete)
                
                - 7 cell type features (all continuous)
        
        - Beta values (450K, 110):  
        
            - Rows: 450K Illumina Infinium HumanMethylation BeadChip in DNA methylation levels.
            
            - Columns: 110 cord blood samples ID
            
        - Data Processing:
        
            - Connect the two datasets above by using limma fit, and we got the relationship between 450K BeadChips and 23 features under the 110 cord blood samples.
            
            - Extract the coefficients of the relationship above.
            
            - Because the datasets above are too large, we shrink it to a ***756 BeadChips*** and 23 features matrix..

    """)

df = pd.read_csv('https://raw.githubusercontent.com/LoWeT0619/CMSE_830_Final_Proj/main/PB_data_coeff.csv')
with col2:
    data = st.button('Data')
if data:
    st.dataframe(df)
st.text("")

# 2 Exploratory Data Analysis (EDA):
st.markdown("### 2 Exploratory Data Analysis (EDA)")
col1, col2 = st.columns(2)
with col1:
    corr_hm = st.checkbox('Check correlations among all features')
if corr_hm:
    st.write('Correlation Heatmap')
    sns.set_theme(style="white")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(25, 25))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    st.pyplot(f)

with col2:
    sign_feat = st.checkbox('Check significant (|correlation| > 0.20) features')
if sign_feat:
    st.markdown("""
    - According the analysis above, our feature candidates associated with preterm birth are:

        - continuous clinical features: **gestationalAge**, **paternalAge**

        - discrete clinical features: **idiopathicPromWithoutInflammation**, **preeclampsia**, **placentaPrevia** 

        - cell type features: **Bcell**, **Cd4t**, **Cd8t**, **Gran**, **Mono**, **Nk**, **Nrbc**
    """)

# 3. Machine Learning Model Pipeline
st.markdown("### 3. Machine Learning Model Pipeline")

## 3.1 Input datasets:
st.markdown("#### 3.1 Input datasets")
col1, col2 = st.columns(2)
with col1:
    X = df[["gestationalAge", "paternalAge", "idiopathicPromWithoutInflammation Yes", "preeclampsia Yes",
            "placentaPrevia Yes", "estimatedBcell", "estimatedCd4t", "estimatedCd8t", "estimatedGran",
            "estimatedMono", "estimatedNk", "estimatedNrbc"]]
    X_data = st.button('X (756 12)')
if X_data:
    st.dataframe(X)
st.text("")

with col2:
    y = df["pretermBirthYes"]
    for i, ele in enumerate(y):
        if ele < 0:
            y[i] = 0
        else:
            y[i] = 1
    y = y.astype('int')
    y_data = st.button('y (756, 1)')
if y_data:
    st.dataframe(y)
st.text("")

## 3.2 Machine Learning Model Choose
st.markdown("#### 3.2 Machine Learning Model Choose")
option = st.selectbox(
    'Which ML Classification Model would you like to explore: ',
    ('Logistic Regression', 'Decision Tree', 'Random Forest',
     'Support Vector Machine (SVM)', 'K-Nearest Neighbour (KNN)', 'Neural Net'))

if option == 'Logistic Regression':
    params = [0.001, 0.01, 0.1, 0.5, 1]
    acc = []
    for param in params:
        clf = LogisticRegression(C=param)
        scores = cross_val_score(clf, X, y, cv=5)
        acc.append(scores.mean())
    st.write('Logistic Regression - Hyperparameter vs Cross Validation Score: ')
    fig, ax = plt.subplots()
    ax.plot(params, acc)
    plt.xlabel("Hyperparameter")
    plt.ylabel("Cross Validation Score")
    st.pyplot(fig)

elif option == 'Decision Tree':
    params = [1, 5, 10, 50, 100]
    acc = []
    for param in params:
        clf = DecisionTreeClassifier(max_depth=param)
        scores = cross_val_score(clf, X, y, cv=5)
        acc.append(scores.mean())
    st.write('Decision Tree - Hyperparameter vs Cross Validation Score: ')
    fig, ax = plt.subplots()
    ax.plot(params, acc)
    plt.xlabel("Hyperparameter")
    plt.ylabel("Cross Validation Score")
    st.pyplot(fig)

elif option == 'Random Forest':
    params = [1, 5, 10, 50, 100]
    acc = []
    for param in params:
        clf = RandomForestClassifier(max_depth=param, n_estimators=10, max_features=1)
        scores = cross_val_score(clf, X, y, cv=5)
        acc.append(scores.mean())
    st.write('Random Forest - Hyperparameter vs Cross Validation Score: ')
    fig, ax = plt.subplots()
    ax.plot(params, acc)
    plt.xlabel("Hyperparameter")
    plt.ylabel("Cross Validation Score")
    st.pyplot(fig)

elif option == 'Support Vector Machine (SVM)':
    params = [0.001, 0.002, 0.005, 0.01, 0.1]
    acc = []
    for param in params:
        clf = SVC(gamma=param)
        scores = cross_val_score(clf, X, y, cv=5)
        acc.append(scores.mean())
    st.write('SVM - Hyperparameter vs Cross Validation Score: ')
    fig, ax = plt.subplots()
    ax.plot(params, acc)
    plt.xlabel("Hyperparameter")
    plt.ylabel("Cross Validation Score")
    st.pyplot(fig)

elif option == 'K-Nearest Neighbour (KNN)':
    params = [1, 2, 4, 5, 10]
    acc = []
    for param in params:
        clf = KNeighborsClassifier(param)
        scores = cross_val_score(clf, X, y, cv=5)
        acc.append(scores.mean())
    st.write('KNN - Hyperparameter vs Cross Validation Score: ')
    fig, ax = plt.subplots()
    ax.plot(params, acc)
    plt.xlabel("Hyperparameter")
    plt.ylabel("Cross Validation Score")
    st.pyplot(fig)

else:
    params = [0.001, 0.01, 0.1, 1, 10]
    acc = []
    for param in params:
        clf = MLPClassifier(alpha=param, max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=5)
        acc.append(scores.mean())
    st.write('Neural Net - Hyperparameter vs Cross Validation Score: ')
    fig, ax = plt.subplots()
    ax.plot(params, acc)
    plt.xlabel("Hyperparameter")
    plt.ylabel("Cross Validation Score")
    st.pyplot(fig)

## 3.3 The best hyperparameter for each model
st.markdown("#### 3.3 The best hyperparameter for each model")
model_list = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine (SVM)', 'K-Nearest Neighbour (KNN)', 'Neural Net']
param_list = [1, 5, 10, 0.1, 10, 2]
results = pd.DataFrame({'Model': model_list, 'Hyperparameter': param_list})
button = st.button('Best Hyperparameter for Each Model')
if button:
    st.dataframe(results)
st.text("")

# 4. Model Evaluation
st.markdown("### 4. Model Evaluation")

## 4.1 Split Dataset into Training and Testing Set
st.markdown("#### 4.1 Split Dataset into Training and Testing Set")
test_size = st.number_input('Insert a number between 0 to 1 as test size', value=0.3, min_value=0.0, max_value=1.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
st.write('The test size is ', test_size)

## 4.2 Model Performance
st.markdown("#### 4.2 Model Performance")

model_pipline = []
model_pipline.append(LogisticRegression(C=param_list[0]))
model_pipline.append(DecisionTreeClassifier(max_depth=param_list[1]))
model_pipline.append(RandomForestClassifier(max_depth=param_list[2], n_estimators=10, max_features=1))
model_pipline.append(SVC(gamma=param_list[3]))
model_pipline.append(KNeighborsClassifier(param_list[4]))
model_pipline.append(MLPClassifier(alpha=param_list[5], max_iter=1000))

model_list = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine (SVM)', 'K-Nearest Neighbour (KNN)', 'Neural Net']
acc_list = []
auc_list = []
cm_list = []

for model in model_pipline:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    fpr, tpr, _thresholds = roc_curve(y_test, y_pred)
    auc_list.append(round(auc(fpr, tpr), 2))
    cm_list.append(confusion_matrix(y_test, y_pred))

col1, col2 = st.columns(2)
with col1:
    cms = st.button('Confusion Matrix')
if cms:
    fig = plt.figure(figsize=(18, 10))
    for i in range(len(cm_list)):
        cm = cm_list[i]
        model = model_list[i]
        sub = fig.add_subplot(2, 3, i+1).set_title(model)
        cm_plot = sns.heatmap(cm, annot=True, cmap='Blues_r')
        cm_plot.set_xlabel('Predicted Values')
        cm_plot.set_ylabel('Ground Truth')
    st.pyplot(fig)
    st.text("")

with col2:
    Acc_Auc = st.button('Accuracy and AUC')
if Acc_Auc:
    results_df = pd.DataFrame({'Model': model_list, 'Accuracy': acc_list, 'AUC': auc_list})
    st.dataframe(results_df)
    st.text("")

# 5. Conclusion and discussion:
st.markdown("### 5. Conclusion and discussion")
col1, col2 = st.columns(2)
with col1:
    conclusion = st.button('Conclusion')
if conclusion:
    st.markdown("""
    - We apply the Machine Learning analysis on 6 Classification Models ('Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine (SVM)', 'K-Nearest Neighbour (KNN)', 'Neural Net'):

    - According the analysis above, the best model is:

        - Random Forest with hyperparameter equals to 10.
    """)

with col2:
    discussion = st.button('Discussion')
if discussion:
    st.markdown("""
    
    - Check overfitting.
    
    - Apply the model to whole dataset, which is 450K.
    """)
st.text("")

st.markdown("[Reference](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501)")