import streamlit as st
import pandas as pd
import numpy as np
import pickle

#Configration Page
st.set_page_config("Customer Churn Prediction",page_icon=':information_desk_person:',layout='wide')
style = "<style>h2 {text-align: center};color=Red"
st.markdown(style,unsafe_allow_html=True)

if 'submited' not in st.session_state:
    st.session_state['submited'] = False
if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False
def submit_button():
    st.session_state.submited = True
def cancel_button():
    st.session_state.submited = False
def predict_button():
    st.session_state['predicted'] = True
def reset_predict():
    st.session_state['predicted'] = False

#Function
def load_model():
    with open('Logistic Regression Churn.sav','rb') as file:
        model = pickle.load(file)
    return model

def predict(data:pd.DataFrame):
    model = load_model()
    prob = model.predict_proba(data)
    return prob[:,1]


#Title and Note
st.title("TLC.Co Customer Churn Prediction")
st.write("""
Welcome to the Customer Churn Prediction Tool.\n
a customer churn app helps businesses understand why customers might leave and allows them to take proactive steps to keep them happy and engaged. This translates to higher customer retention, increased revenue, and a more sustainable business model.
"""
)
st.divider()

# with st.sidebar:
#     st.header("Menu",divider='gray')
#     st.button("Home",use_container_width=True)
#     st.button('Setting',use_container_width=True)
#     st.button("About",use_container_width=True)


#Main Pages
#Membuat dua kolom
left_panel,right_panel = st.columns(2, gap='medium')
#Left Panel
left_panel.header('Information Panel')
#Membuat Tabs Overviews di left Panel
tabs1,tabs2,tabs3 = left_panel.tabs(['Overview','Analytic Approach','Model Summary'])
#Tabs1
tabs1.subheader('Overview')
tabs1.write("""
Customer churn apps are tools that help businesses understand and reduce customer churn, which is the rate at which customers stop using a product or service. Here's a quick rundown:
""")
tabs1.write("""
Why are Customer Churn Apps Important?

1. High churn rates can stifle growth and revenue. Acquiring new customers is typically more expensive than retaining existing ones.
2. Customer churn apps help identify at-risk users and understand the reasons behind churn.
""")
tabs1.write("""
What do Customer Churn Apps Do?

1. Track user behavior and engagement metrics to identify users who are disengaging.
2. Analyze user data to pinpoint potential causes of churn, like how long a customer has been with us and their contract type.
3. May enable targeted interventions to win back at-risk users, like personalized messages or exclusive offers.
""")
tabs1.write("""
Benefits of Customer Churn Apps

1. Improved customer retention rates
2. Increased customer lifetime value
3. Better understanding of user needs and preferences""")

#Tabs2
tabs2.subheader('Analytic Approach')
tabs2.write("""
We will first analyze the data to identify patterns that differentiate unsubscribing customers from those who remain subscribed. Then we will build a classification model to predict the probability of a customer unsubscribing.
\nTo assess a classification model's ability to predict different classes, we can use a confusion matrix, as shown in the example below.
""")
tabs2.write("""
|  | Actual Positive | Actual Negative |
|---|---|---|
| Predicted Positive | True Positve | False Positve |
| Predicted Negative | False Negative | True Negative |
""")
tabs2.write("\n")
tabs2.write("""
\nImagine a scorecard that shows how well a system predicts customer churn (who will leave). There are two mistakes it can make:
\n1. Wrong churn (False Positive): The system says someone will leave (churn) but they actually stay. This wastes resources because we try to win them back when we don't need to.
\n2. Missed churn (False Negative): The system says someone will stay but they actually leave. This means we lose money because we miss the chance to save them.
\nThere's no perfect system, reducing one mistake often increases the other. So we need to find a good balance between the two.We want to build a system to stop customers from leaving (churning).
""")
tabs2.write("""
\nIdeally, we only target customers who actually will leave, not waste effort on those staying.
It's more important to catch real churners (even if we target a few who stay accidentally) because missing them means losing money.
So, to measure how well we're doing, we focus on a score called "recall" which shows how good we are at finding true churners.
""")

#Tabs 3
tabs3.subheader("Model Summary")
tabs3.write("""
We used Logistic Regression model because it's strikes a good balance between strong performance and practical advantages, making it the optimal choice for this customer churn prediction task.
""")
logreg, eval, oddr, feature_imp = tabs3.tabs(['Model','Evaluation','Odds Ratio','Feature Importance'])
logreg.write("""
Logistic regression is a powerful tool for predicting customer churn. Here's a breakdown of its magic:
""")
col1, col2, col3 = logreg.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image('Persamaan Sigmoid.png', caption='Graph of Sigmoid Equation',use_column_width=True)
with col3:
    st.write(' ')
logreg.write("""
1. Learning the Curve: Imagine a flexible curve that fits the churn patterns in your data. Logistic regression creates this S-shaped curve, called a sigmoid function. This curve helps estimate the probability of churn for each customer.
\n2. From Data to Probability: Each customer's data (purchase history, demographics, etc.) is fed into the equation. The sigmoid function then crunches the numbers and spits out a probability between 0 and 1.
\n3. Setting the Bar (Threshold): Not every customer with a churn probability needs immediate action. We set a threshold, like 50%. Customers exceeding this threshold (high churn probability) are flagged for potential intervention.
""")

eval.write("""
Model evaluation is crucial for assessing the generalizability of our model. This process involves testing the model's performance on unseen data to identify potential overfitting.
""")
eval.write("\n")
eval.write("""
|Class| Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
|0| 0.92 | 0.62 | 0.74 | 713 |
|1| 0.45 | 0.85 | 0.59 | 258 |
""")
eval.write("""
\nOur logistic regression model demonstrates promising results in predicting customer churn on unseen testing data. This is evidenced by its high recall value of 0.85. In simpler terms, for every 100 actual churning customers in new real-world data, the model can correctly identify 85 of them.""")
eval.image('ROC AUC.png', caption='ROC AUC plot',use_column_width=True)
eval.write("""
The logistic regression model achieved a ROC AUC of 0.83 on the testing data. This value falls between 0.5 and 1, indicating good discrimination between customer churn (class 1) and non-churn (class 0). In simpler terms, the model performs better than random guessing at distinguishing between these two classes.
""")

oddr.write("""
Odds Ratio compares the odds of something happening in one situation (Situation A) to the odds of it happening in another situation (Situation B). For our customer churn prediction this is the breakdown of each feature that used.
""")
oddr.image('Odds Ratio.png', caption='Table of Odds Ratio Each Feature',use_column_width=True)

feature_imp.write("""
Feature importance is like figuring out which ingredients matter most in a recipe. In machine learning, it's about understanding how much each piece of data (called a feature) contributes to a model's predictions.
""")
cols1, cols2 =feature_imp.columns(2)
with cols1:
    st.image('Feature Importance.png', caption='Magnitude of Feature Importance',use_column_width=True)    
with cols2:
    st.image('Feature Importance directions.png', caption='Directions of Feature Importance',use_column_width=True)
feature_imp.write("\n")
feature_imp.write("""
|Feature|the higher the value|the lower the value|
|---|---|---|
|Dependents|the lower the probailiy churn|the higher the probailiy churn|
|Online Security|the lower the probailiy churn|the higher the probailiy churn|
|Internet Service Fiber Optic|the higher the probailiy churn|the lower the probailiy churn|
|No Internet Service|the lower the probailiy churn|the higher the probailiy churn|
|Contract Month to month|the higher the probailiy churn|the lower the probailiy churn|
|Contract Two Year|the lower the probailiy churn|the higher the probailiy churn|
|Paperless Billing|the lower the probailiy churn|the higher the probailiy churn|
|Tenure|the lower the probailiy churn|the higher the probailiy churn|
""")
#Right Panel
right_panel.header('Prediction')
placeholder = right_panel.empty()
input_container = placeholder.container()
right_panel.divider()
result_placeholder = right_panel.empty()
result_container = result_placeholder.container()
btn_placeholder = right_panel.empty()

cust_id = input_container.text_input('Customer ID :', label_visibility='collapsed',placeholder='Customer ID :')
input_container.write("\n")
left_input, right_input = input_container.columns(2)
#Left Feature
left_input.write("**Customer Information**")
tenure = left_input.number_input('Tenure (in Month)', min_value=0, step=1)
Dependents = left_input.selectbox("Have dependents?", options=["Yes","No"])
Contract =left_input.selectbox("Customer Contract", options=["Month-to-month","One Year","Two Year"])
PaperlessBilling = left_input.selectbox("Paperless Billing", options=["Yes","No"])
MonthlyCharges = left_input.number_input('Monthly Charges (in $)', min_value=1, step=10)

#Right Feature
right_input.write("**Customer Product**")
OnlineSecurity = right_input.selectbox("Using Online Security?", options=["Yes","No"])
OnlineBackup = right_input.selectbox("Using Online Backup?", options=["Yes","No"])
InternetService = right_input.selectbox("Using Internet Service?", options=["DSL","Fiber optic","No"])
DeviceProtection = right_input.selectbox("Using Device Protection?", options=["Yes","No"])
TechSupport = right_input.selectbox("Using Tech Support?", options=["Yes","No"])

data_list = {"Personal Information":["Customer ID",'tenure',"Dependents","OnlineSecurity",'OnlineBackup','InternetService','DeviceProtection','TechSupport','Contract','PaperlessBilling','MonthlyCharges'],
        "Value":[cust_id,tenure,Dependents, OnlineSecurity,OnlineBackup, InternetService, DeviceProtection,TechSupport,Contract,PaperlessBilling,MonthlyCharges]}

#Submit Button
btn_submit = btn_placeholder.button('Submit',use_container_width=True, on_click=submit_button)

if st.session_state['submited']:
    placeholder.dataframe(data_list, use_container_width=True)
    btn_after =  btn_placeholder.container()
    btn_predict = btn_after.button('Predict',use_container_width=True)
    btn_cancel = btn_after.button('Cancel',use_container_width=True,on_click=cancel_button)
    if btn_predict:
        data = {'tenure':tenure,"Dependents":Dependents,"OnlineSecurity":OnlineSecurity,'OnlineBackup':OnlineBackup,'InternetService':InternetService,'DeviceProtection':DeviceProtection,'TechSupport':TechSupport,'Contract':Contract,'PaperlessBilling':PaperlessBilling,'MonthlyCharges':MonthlyCharges}
        data = pd.DataFrame(data,index=[1])
        if predict(pd.DataFrame(data))[0] <= 0.5:
            result_container.success(f"Customer with ID: {cust_id} have {predict(pd.DataFrame(data))[0].round(2)*100}% probability to churn.")
            right_panel.balloons()
            result_container.write("Customers are likely to stay loyal. Keep your loyal customers engaged by offering them loyalty program")
        if predict(pd.DataFrame(data))[0] >= 0.5:
            result_container.error(f"Customer with ID: {cust_id} have {predict(pd.DataFrame(data))[0].round(2)*100}% probability to churn.")
            # right_panel.balloons()
            result_container.write("Customers are likely to churn. Offer targeted incentives to regain churned customers.")
        btn_predict_again = btn_placeholder.button('Predict Again',use_container_width=True,on_click=cancel_button)