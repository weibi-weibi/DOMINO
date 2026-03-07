import os
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(layout="wide")
st.title("TGAT & G Model Dashboard")


# ----------------------------------------------------
# Load Data
# ----------------------------------------------------
DATA_DIR =  r'C:\Users\w.bi\OneDrive - IESEG\Desktop\Domino\data'
@st.cache_data
def load_data(path, mtime):
    return pd.read_csv(path)


# ----------------------------------------------------
# Sidebar Filters
# ----------------------------------------------------
st.sidebar.header("Filters")

model_options = ['All Reasons', 'Crossing', 'Blocking']
selected_model = st.sidebar.selectbox("Select Model", model_options)

if selected_model == 'All Reasons':
    file_path = os.path.join(DATA_DIR, 'TGAT', '1stage', 'final_predictions.csv')
    file_modified_time = os.path.getmtime(file_path)
elif selected_model == 'Crossing':
    file_path = os.path.join(DATA_DIR, 'TGAT', '1stage', 'final_predictions_c1134.csv')
    file_modified_time = os.path.getmtime(file_path)
else:
    file_path = os.path.join(DATA_DIR, 'TGAT', '1stage', 'final_predictions_c1137.csv')
    file_modified_time = os.path.getmtime(file_path)

df = load_data(file_path, file_modified_time)



month_options = sorted(df["month"].unique())
selected_month = st.sidebar.selectbox("Select Month", month_options)
df_month = df[df["month"] == selected_month]

date_options = sorted(df_month["date"].unique())
selected_date = st.sidebar.selectbox("Select Date", date_options)
df_date = df_month[df_month["date"] == selected_date]

hour_options = sorted(df_date["hour"].unique())
selected_hour = st.sidebar.selectbox("Select Time (Hour)", hour_options)
df_hour = df_date[df_date["hour"] == selected_hour]

front_options = sorted(df_hour["j"].unique())
selected_front = st.sidebar.selectbox("Select Front Train", front_options)
df_front = df_hour[df_hour["j"] == selected_front]


# checkpoint_options = sorted(df_front["j_checkpt_name"].unique())
# selected_checkpoint = st.sidebar.selectbox("Select Checkpoint", checkpoint_options)
# df_checkpoint = df_front[df_front["j_checkpt_name"] == selected_checkpoint]

checkpoint_options = ["All"] + sorted(df_front["j_checkpt_name"].unique())

selected_checkpoint = st.sidebar.selectbox(
    "Select Checkpoint",
    checkpoint_options
)

if selected_checkpoint == "All":
    df_checkpoint = df_front
else:
    df_checkpoint = df_front[df_front["j_checkpt_name"] == selected_checkpoint]


# cause_options = sorted(df_checkpoint["delay_cause_fr"].unique())
# selected_cause = st.sidebar.selectbox("Select Delay Reason", cause_options)
# df_cause = df_checkpoint[df_checkpoint["delay_cause_fr"] == selected_cause]


filtered_df = df_checkpoint



# -----------------------------
# G predicted-class confidence
# -----------------------------
g_prob_cols = ["G_prob_0", "G_prob_1", "G_prob_2", "G_prob_3"]

tgat_prob_cols = [
    "TGAT_prob_0", 
    "TGAT_prob_1", 
    "TGAT_prob_2", 
    "TGAT_prob_3"
]

def safe_multiclass_roc(df, prob_cols, label_col="y_multi"):
    try:
        if df[label_col].nunique() < 2:
            return None
        return roc_auc_score(
            df[label_col],
            df[prob_cols],
            multi_class="ovr",
            average="macro"
        )
    except:
        return None

roc_values = {
    "Month": df_month,
    "Day": df_date,
    # "Hour": df_hour,
    # "Train j": df_front
}

roc_results = {}

for level, df in roc_values.items():

    g_roc = safe_multiclass_roc(df, g_prob_cols)
    tgat_roc = safe_multiclass_roc(df, tgat_prob_cols)

    roc_results[level] = {
        "G": g_roc,
        "TGAT": tgat_roc
    }

st.subheader("ROC-AUC Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### G Model")
    for level in roc_results:
        val = roc_results[level]["G"]
        st.metric(level, f"{val:.3f}" if val else "N/A")

with col2:
    st.markdown("### TGAT Model")
    for level in roc_results:
        val = roc_results[level]["TGAT"]
        st.metric(level, f"{val:.3f}" if val else "N/A")

# roc_month = safe_multiclass_roc(df_month, g_prob_cols, "y_multi")
# roc_day   = safe_multiclass_roc(df_date, g_prob_cols, "y_multi")
# roc_hour  = safe_multiclass_roc(df_hour, g_prob_cols, "y_multi")
# roc_train = safe_multiclass_roc(df_front, g_prob_cols, "y_multi")

# st.subheader("ROC-AUC Performance")

# col1, col2, col3, col4 = st.columns(4)

# col1.metric("Month ROC-AUC", f"{roc_month:.3f}" if roc_month else "N/A")
# col2.metric("Day ROC-AUC", f"{roc_day:.3f}" if roc_day else "N/A")
# col3.metric("Hour ROC-AUC", f"{roc_hour:.3f}" if roc_hour else "N/A")
# col4.metric("Train ROC-AUC", f"{roc_train:.3f}" if roc_train else "N/A")


# filtered_df["G_confidence"] = (
#     filtered_df[g_prob_cols]
#     .to_numpy()[np.arange(len(filtered_df)), 
#                 filtered_df["G_multi"].astype(int)]
#     * 100
# ).round(1)

filtered_df["G_probabilities"] = (
    filtered_df[g_prob_cols]
    .apply(lambda x: "<br>".join([f"Class {i}: {p*100: .1f}" for i, p in enumerate(x)]), axis=1)
)

# -----------------------------
# TGAT predicted-class confidence
# -----------------------------


# filtered_df["TGAT_confidence"] = (
#     filtered_df[tgat_prob_cols]
#     .to_numpy()[np.arange(len(filtered_df)), 
#                 filtered_df["TGAT_multi_new"].astype(int)]
#     * 100
# ).round(1)

filtered_df["TGAT_probabilities"] = (
    filtered_df[tgat_prob_cols]
    .apply(lambda x: "<br>".join([f"Class {i}: {p*100: .1f}" for i, p in enumerate(x)]), axis=1)
)

st.subheader("Filtered Prediction Results")

st.write("Secondary Delay Classification")
st.write("Class 0: no delay; Class 1: 60 - 299 seconds; Class 2: 300 - 599 seconds; Class 3: >600 seconds. ")
st.write(f"Number of Events: {len(filtered_df)}")

display_cols = [
    # "cause",
    "i",
    "i_checkpt_name",
    "time",
    "G_sum",
    "y_multi",
    "G_multi",
    "G_probabilities",
    "TGAT_multi_new",
    "TGAT_probabilities",
    "delay_cause_fr",
    # "delay_cause_nl"
]

table_df = (
    filtered_df[display_cols]
    .sort_values("G_sum", ascending=False)
    .rename(columns={
        "i": "Train Affected",
        "i_checkpt_name": "Train Affected's Location",
        "time": "Time Interval",     
        "delay_cause_fr": "Delay Cause",
        "G_sum": "Resource Share Level",
        "y_multi": "True Class",
        "G_multi": "G Prediction",
        "G_probabilities": "G Probabilities (%)",
        "TGAT_multi_new": "TGAT Prediction",
        "TGAT_probabilities": "TGAT Probabilities (%)"
    })
    .T
)


html_table = table_df.to_html(escape=False)

st.markdown("""
<style>
.table-scroll {
    overflow-x: auto;   /* keep horizontal scroll */
    width: 100%;
}

.table-scroll table {
    border-collapse: collapse;
}

.table-scroll th, .table-scroll td {
    padding: 6px 10px;
    text-align: center;
    white-space: nowrap;   /* prevents width jumping */
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="table-scroll">{html_table}</div>',
    unsafe_allow_html=True
)

# st.markdown(
#     table_df.to_html(escape=False),
#     unsafe_allow_html=True
# )


