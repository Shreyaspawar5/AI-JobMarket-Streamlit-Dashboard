# ai_jobs_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from itertools import chain
st.set_page_config(layout="wide")

# Load Data
df = pd.read_csv("ai_job_dataset.csv")
df.columns = df.columns.str.lower().str.strip()
df["posting_date"] = pd.to_datetime(df["posting_date"])
df["application_deadline"] = pd.to_datetime(df["application_deadline"])
categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: col.str.strip())
df = df.dropna()

# Derived columns
df['work_type'] = df['remote_ratio'].apply(lambda x: 'Remote' if x >= 50 else 'Onsite/Hybrid')
exp_level_map = {
    "EN": "Entry-level",
    "MI": "Mid-level",
    "SE": "Senior-level",
    "EX": "Executive-level"
}
df["experience_level_full"] = df["experience_level"].map(exp_level_map).fillna(df["experience_level"])

st.title("📊 AI Jobs Interactive Dashboard")
st.markdown("Explore salary trends, skill demand, and hiring patterns in the global AI job market.")

# Sidebar filters
with st.sidebar:
    st.header("🔎 Filters")
    selected_countries = st.multiselect("Company Location", options=sorted(df["company_location"].unique()))
    selected_experience = st.multiselect("Experience Level", options=list(exp_level_map.values()))

    if selected_countries:
        df = df[df["company_location"].isin(selected_countries)]
    if selected_experience:
        rev_exp_map = {v: k for k, v in exp_level_map.items()}
        selected_codes = [rev_exp_map[e] for e in selected_experience]
        df = df[df["experience_level"].isin(selected_codes)]

# Theme
sns.set_theme(style="whitegrid", palette="rocket")
plot_width, plot_height = 7, 4

# Tabs
tabs = st.tabs([
    "🌍 Salary by Country",
    "💼 Top Job Titles",
    "🏭 Industries Hiring",
    "🧠 Skills & Wordcloud",
    "📈 Education Level Benefits",
    "📅 Top Hiring Companies",
    "📊 Work Type Distribution"
])

with tabs[0]:
    st.subheader("Average AI Job Salary by Country")
    fig = px.choropleth(df.groupby("company_location")["salary_usd"].mean().reset_index(),
                        locations="company_location",
                        locationmode="country names",
                        color="salary_usd",
                        color_continuous_scale="Inferno")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Top 10 AI Job Titles by Avg Salary")
    top_roles = df.groupby("job_title")["salary_usd"].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    sns.barplot(x=top_roles.values, y=top_roles.index, ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Top Hiring Industries")
    top_ind = df["industry"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    sns.barplot(x=top_ind.values, y=top_ind.index, ax=ax)
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Top 20 Required Skills")
    skills = df["required_skills"].str.split(", ")
    skill_counts = Counter(chain.from_iterable(skills))
    common_skills = pd.DataFrame(skill_counts.most_common(20), columns=["Skill", "Frequency"])
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    sns.barplot(data=common_skills, y="Skill", x="Frequency", ax=ax)
    st.pyplot(fig)

    st.subheader("Word Cloud of Required Skills")
    flat = ", ".join(df["required_skills"].dropna())
    wc = WordCloud(width=800, height=400, background_color='white', colormap='rocket').generate(flat)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

with tabs[4]:
    st.subheader("Avg Benefits Score by Education Level")
    edu_benefits = df.groupby("education_required")["benefits_score"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    sns.barplot(x=edu_benefits.values, y=edu_benefits.index, ax=ax)
    for i, v in enumerate(edu_benefits.values):
        ax.text(v + 0.05, i, f"{v:.2f}", va='center')
    st.pyplot(fig)

with tabs[5]:
    st.subheader("Top Hiring Companies")
    top_companies = df["company_name"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    sns.barplot(x=top_companies.values, y=top_companies.index, ax=ax)
    st.pyplot(fig)

with tabs[6]:
    st.subheader("Work Type Distribution by Country")
    work = df.groupby(['employee_residence', 'work_type']).size().unstack().fillna(0)
    work = work[work.sum(axis=1) > 20].sort_values(by='Remote', ascending=False)
    fig, ax = plt.subplots(figsize=(plot_width, plot_height*2))
    work.plot(kind='barh', stacked=True, ax=ax, colormap='rocket')
    st.pyplot(fig)

st.markdown("---")
st.caption("AI Job Trends 2025")

