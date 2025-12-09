import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

# Title and disclaimer
st.title("LinkedIn User Prediction App")
st.write("Predicts LinkedIn usage based on demographic information using machine learning")
st.caption("Marketing Analytics Team | Data from Pew Research Center | To be used for educational purposes only")
st.write("---")

# Load data
@st.cache_data
def load_data():
    s = pd.read_csv('social_media_usage.csv')
    
    def clean_sm(x):
        return np.where(x == 1, 1, 0)
    
    ss = s[['web1h', 'income', 'educ2', 'par', 'marital', 'gender', 'age']].copy()
    ss['sm_li'] = clean_sm(ss['web1h'])
    ss['parent'] = np.where(ss['par'] == 1, 1, 0)
    ss['married'] = np.where(ss['marital'] == 1, 1, 0)
    ss['female'] = np.where(ss['gender'] == 2, 1, 0)
    ss['income'] = np.where(ss['income'] > 9, np.nan, ss['income'])
    ss['educ2'] = np.where(ss['educ2'] > 8, np.nan, ss['educ2'])
    ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age'])
    ss = ss.dropna()
    ss = ss[['sm_li', 'income', 'educ2', 'parent', 'married', 'female', 'age']]
    return ss

# Train model
@st.cache_resource
def train_model(data):
    y = data['sm_li']
    X = data[['income', 'educ2', 'parent', 'married', 'female', 'age']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    return lr

ss = load_data()
model = train_model(ss)

# Inputs
st.subheader("Enter Your Information")

st.write("**Household Income:**")
income_options = {
    1: "Less than $10,000",
    2: "$10,000 - $19,999",
    3: "$20,000 - $29,999",
    4: "$30,000 - $39,999",
    5: "$40,000 - $49,999",
    6: "$50,000 - $74,999",
    7: "$75,000 - $99,999",
    8: "$100,000 - $149,999",
    9: "$150,000 or more"
}

income = st.selectbox(
    "Select your household income range",
    options=list(income_options.keys()),
    format_func=lambda x: income_options[x],
    index=4
)

st.write("**Education Level:**")
education_options = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Associate degree",
    6: "Bachelor's degree",
    7: "Some graduate school",
    8: "Graduate/Professional degree (MA, PhD, MD, JD)"
}

education = st.selectbox(
    "Select your highest education level",
    options=list(education_options.keys()),
    format_func=lambda x: education_options[x],
    index=3
)

parent = st.radio("Are you a parent of a child under 18?", ["No", "Yes"])
married = st.radio("Are you married?", ["No", "Yes"])
gender = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 97, 42)

# Convert to numbers
if parent == "Yes":
    parent_val = 1
else:
    parent_val = 0

if married == "Yes":
    married_val = 1
else:
    married_val = 0

if gender == "Female":
    female_val = 1
else:
    female_val = 0

# Predict button
if st.button("Predict LinkedIn Usage"):
    
    # Make dataframe with inputs
    new_data = pd.DataFrame({
        'income': [income],
        'educ2': [education],
        'parent': [parent_val],
        'married': [married_val],
        'female': [female_val],
        'age': [age]
    })
    
    # Get prediction
    pred = model.predict(new_data)[0]
    prob = model.predict_proba(new_data)[0][1]
    
    # Show result
    st.write("---")
    st.subheader("Prediction Results")
    
    if pred == 1:
        st.success("Predicted: LinkedIn User")
    else:
        st.warning("Predicted: Not a LinkedIn User")
    
    st.metric("Probability of LinkedIn Usage", f"{prob:.1%}")
    
    # Make a chart
    prob_data = pd.DataFrame({
        'Result': ['Not User', 'User'],
        'Probability': [1-prob, prob]
    })
    
    fig = px.bar(prob_data, x='Result', y='Probability', 
                 title='Prediction Probabilities',
                 color='Result',
                 color_discrete_map={'Not User': 'lightcoral', 'User': 'lightblue'})
    fig.update_yaxes(range=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: Hover over the chart to see values. Click the camera icon to download as PNG.")
    
    # Compare to average
    st.write("---")
    st.subheader("Profile Comparison")
    st.write("How does this profile compare to the average LinkedIn user in our dataset?")
    
    linkedin_users = ss[ss['sm_li'] == 1]
    avg_income = linkedin_users['income'].mean()
    avg_education = linkedin_users['educ2'].mean()
    avg_age = linkedin_users['age'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Your Profile:**")
        st.write(f"Income Level: {income}")
        st.write(f"Education Level: {education}")
        st.write(f"Age: {age}")
    
    with col2:
        st.write("**Average LinkedIn User:**")
        st.write(f"Income Level: {avg_income:.1f}")
        st.write(f"Education Level: {avg_education:.1f}")
        st.write(f"Age: {avg_age:.1f}")
    
    # Make comparison chart
    compare = pd.DataFrame({
        'Variable': ['Income', 'Education', 'Age'],
        'Your Profile': [income, education, age],
        'LinkedIn Average': [avg_income, avg_education, avg_age]
    })
    
    fig2 = px.bar(compare, x='Variable', y=['Your Profile', 'LinkedIn Average'], 
                  barmode='group', title='Your Profile vs Average LinkedIn User')
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Tip: Hover over the chart to see values. Click the camera icon to download as PNG.")
    
    # Sweet Spot Demographics
    st.write("---")
    st.subheader("Who Are the Most Likely LinkedIn Users?")
    st.write("Based on our data, here are the demographic profiles most likely to use LinkedIn:")
    
    high_prob_users = ss[ss['sm_li'] == 1]
    low_prob_users = ss[ss['sm_li'] == 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Most Likely to Use LinkedIn:**")
        st.metric("Average Income Level", f"{high_prob_users['income'].mean():.1f}")
        st.metric("Average Education Level", f"{high_prob_users['educ2'].mean():.1f}")
        st.metric("Average Age", f"{high_prob_users['age'].mean():.0f}")
    
    with col2:
        st.write("**Least Likely to Use LinkedIn:**")
        st.metric("Average Income Level", f"{low_prob_users['income'].mean():.1f}")
        st.metric("Average Education Level", f"{low_prob_users['educ2'].mean():.1f}")
        st.metric("Average Age", f"{low_prob_users['age'].mean():.0f}")
    
    st.write("**Marketing Insight:** Focus campaigns on higher income, higher education demographics for best ROI on LinkedIn advertising.")
    
    # Marketing insight
    st.write("---")
    st.subheader("Marketing Insights")
    if prob > 0.7:
        st.info("High Priority Target: This demographic profile shows a strong likelihood of LinkedIn usage. Recommended for targeted LinkedIn marketing campaigns.")
    elif prob > 0.5:
        st.info("Moderate Priority Target: This profile shows some potential for LinkedIn engagement. Consider including in broader digital marketing strategies.")
    else:
        st.info("Alternative Platforms Recommended: This demographic may be better reached through other social media channels or marketing platforms.")

# Footer
st.write("---")
st.caption("Built with Streamlit and scikit-learn | Model trained on 1,260 responses | Accuracy: ~66%")