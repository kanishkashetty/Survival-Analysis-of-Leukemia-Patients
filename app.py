import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

# Load your data (you can replace this with your real dataset)
df = pd.read_csv("Standard Survival Data Format.csv") 
df = df.rename(columns={
    'Patient (i)': 'Patient',
    'Remission Time (t)': 'Remission_Time',
    'Censoring (d)': 'Event',
    'Group (X)': 'Group'
})
# Flip group if needed (0 = better group, 1 = worse group)
df['Group_Flip'] = df['Group'].apply(lambda x: 0 if x == 1 else 1)

# Fit Kaplan-Meier models
kmf1 = KaplanMeierFitter()
kmf2 = KaplanMeierFitter()

mask1 = df['Group_Flip'] == 0
mask2 = df['Group_Flip'] == 1

kmf1.fit(df[mask1]['Remission_Time'], df[mask1]['Event'], label='Group 1 (Better)')
kmf2.fit(df[mask2]['Remission_Time'], df[mask2]['Event'], label='Group 2 (Worse)')

# Fit Cox model
cph = CoxPHFitter()
cph.fit(df[['Remission_Time', 'Event', 'Group_Flip']], duration_col='Remission_Time', event_col='Event')

# Predict survival curves for both groups
group_0 = pd.DataFrame({'Group_Flip': [0]})
group_1 = pd.DataFrame({'Group_Flip': [1]})
surv_0 = cph.predict_survival_function(group_0)
surv_1 = cph.predict_survival_function(group_1)

# Streamlit dashboard
st.title("Leukemia Patient Survival Analysis Dashboard")

st.markdown("""
### Kaplan-Meier Survival Curves
This plot shows the observed survival probabilities over time for both groups.
""")
fig, ax = plt.subplots()
kmf1.plot(ax=ax)
kmf2.plot(ax=ax)
plt.title("Kaplan-Meier Survival Curves")
plt.xlabel("Time (weeks)")
plt.ylabel("Survival Probability")
st.pyplot(fig)

st.markdown("""
### Cox Model: Predicted Survival Curves
This plot shows the model-based predictions of survival over time.
""")
fig2, ax2 = plt.subplots()
ax2.plot(surv_0.index, surv_0.values, label='Group 1 (Better)', color='blue')
ax2.plot(surv_1.index, surv_1.values, label='Group 2 (Worse)', color='orange')
ax2.set_title("Cox Model Predicted Survival")
ax2.set_xlabel("Time (weeks)")
ax2.set_ylabel("Survival Probability")
ax2.legend()
st.pyplot(fig2)

st.markdown("""
### Cox Model Summary
Below is the statistical summary of the Cox Proportional Hazards model.
""")
st.text(cph.summary.to_string())
