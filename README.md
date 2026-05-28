# Email Marketing Dashboard

A Streamlit data app for monitoring and A/B test analysis of email marketing campaigns.

## Features

**Monitoring**
- Key metrics tracking (Open Rate, CTR, Open-to-Click, Paid Spend Rate)
- Interactive time-series chart with alert threshold
- Rule × Response heatmaps
- AI-generated executive summary (Groq)

**A/B Analysis**
- Chi-square significance testing across 4 experiment groups
- Results segmented by All / Buyer / Not Buyer
- Lift visualization by metric and segment
- AI-generated recommendation (Groq)

## Data
Email marketing dataset (~472k rows, October 2024) loaded from Google Drive via service account.

## Stack
Python · Streamlit · Plotly · Pandas · SciPy · Groq API · Google Drive API

## Live App
[data-app-hvdfjdoecoz77dlzetu6g7.streamlit.app](https://data-app-hvdfjdoecoz77dlzetu6g7.streamlit.app)
