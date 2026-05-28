import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from groq import Groq

st.set_page_config(page_title="Email Marketing Dashboard", layout="wide")

# ── Synthetic data ──────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=credentials)
    file_id = "18ONnHkUCXaHiDrTGgAONpreX1FdOXeL5"
    request = service.files().get_media(fileId=file_id)
    content = io.BytesIO(request.execute())
    df = pd.read_csv(content, sep=";")

    for col in ['delivery_ts', 'send_ts', 'read_ts', 'click_ts']:
        df[col] = pd.to_datetime(pd.to_numeric(df[col], errors='coerce'), unit='s', utc=True)
    for col in ['last_answer_timestamp', 'confirm_timestamp']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['response'] = df['response'].str.strip().str.lower()
    df['buyer'] = df['buyer'].str.strip()
    df['rule'] = df['rule'].str.strip()
    df['not_free_credits'] = pd.to_numeric(df['not_free_credits'], errors='coerce').fillna(0)
    df['total_credits'] = pd.to_numeric(df['total_credits'], errors='coerce').fillna(0)

    df['is_buyer'] = (df['buyer'] == 'Buyer').astype(int)
    df['is_delivered'] = df['delivery_ts'].notna().astype(int)
    df['is_read'] = df['read_ts'].notna().astype(int)
    df['is_clicked'] = df['click_ts'].notna().astype(int)
    df['is_paid_spend'] = (df['not_free_credits'] > 0).astype(int)

    return df

df = load_data()

# Groq Api
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    groq_client = None

# Sidebar
st.sidebar.title("Filters")
tab_choice = st.sidebar.radio("Section", ["📈 Monitoring", "🧪 A/B Analysis"])

# MONITORING
if tab_choice == "📈 Monitoring":
    st.title("📈 Email Monitoring")

    # Filters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        segment = st.selectbox("Segment", ["All", "Buyer", "Not Buyer"])
    with col2:
        rule = st.selectbox("Rule", ["All"] + sorted(df["rule"].unique().tolist()))
    with col3:
        response = st.selectbox("Response", ["All"] + sorted(df["response"].unique().tolist()))
    with col4:
        threshold = st.slider("Alert threshold (drop %)", 5, 30, 15)

    # Apply filters
    df_sub = df.copy()
    if segment != "All":
        df_sub = df_sub[df_sub["buyer"] == segment]
    if rule != "All":
        df_sub = df_sub[df_sub["rule"] == rule]
    if response != "All":
        df_sub = df_sub[df_sub["response"] == response]

    # Daily metrics
    daily = df_sub.groupby("date").agg(
        sends=("is_read", "count"),
        opens=("is_read", "sum"),
        clicks=("is_clicked", "sum"),
        deliveries=("is_delivered", "sum"),
        spends=("is_paid_spend", "sum")
    ).reset_index()

    daily["avg_not_free_credits"] = (
        df_sub[df_sub["is_clicked"] == 1]
        .groupby("date")["not_free_credits"]
        .mean()
        .reindex(daily["date"])
        .values
    )

    daily["delivery_rate"] = daily["deliveries"] / daily["sends"]
    daily["open_rate"]  = daily["opens"]  / daily["deliveries"]
    daily["ctr"]        = daily["clicks"] / daily["deliveries"]
    daily["open_to_click"] = daily["clicks"] / daily["opens"].replace(0, np.nan)
    daily["paid_spend_rate"] = daily["spends"] / daily["sends"]
    daily["click_to_spend"] = daily["spends"] / daily["clicks"]

    #KPIs
    st.subheader("KPIs")

    daily_kpi = daily[daily["date"].dt.date < daily["date"].dt.date.max()]

    last_date = daily_kpi["date"].iloc[-1].strftime('%b %d')
    prev_date = daily_kpi["date"].iloc[-2].strftime('%b %d') if len(daily_kpi) >= 2 else last_date
    st.caption(f"Delta: {last_date} vs {prev_date}")

    kpi_metrics = {
        "Open Rate": "open_rate",
        "CTR": "ctr",
        "Open-to-Click": "open_to_click",
        "Paid Spend Rate": "paid_spend_rate",
    }

    cols = st.columns(4)
    for col, (label, key) in zip(cols, kpi_metrics.items()):
        last_val = daily_kpi[key].iloc[-1]
        prev_val = daily_kpi[key].iloc[-2] if len(daily_kpi) >= 2 else last_val
        delta = last_val - prev_val
        col.metric(label, f"{last_val:.2%}", f"{delta:+.2%}")

    # Charts
    st.subheader("Metrics over time")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        metric_choice = st.selectbox("Metric", ["delivery_rate", "open_rate", "ctr", "open_to_click", "paid_spend_rate",
                                                "click_to_spend", "avg_not_free_credits"])
    with chart_col2:
        date_min = df_sub["date"].min().date()
        date_max = df_sub["date"].max().date()
        date_range = st.date_input("Date range", [date_min, date_max], min_value=date_min, max_value=date_max)

    daily_chart = daily.copy()
    if len(date_range) == 2:
        daily_chart = daily_chart[
            (daily_chart["date"].dt.date >= date_range[0]) &
            (daily_chart["date"].dt.date <= date_range[1])
            ]

    mean_val = daily[metric_choice].mean()
    alert_val = mean_val * (1 - threshold / 100)

    fig = go.Figure()

    # Sends first — renders behind
    fig.add_trace(go.Bar(
        x=daily_chart["date"], y=daily_chart["sends"],
        name="Sends", marker_color="#D0E4F7", opacity=0.4,
        yaxis="y1", hoverinfo="skip"
    ))

    # Metric line on top
    fig.add_trace(go.Scatter(
        x=daily_chart["date"], y=daily_chart[metric_choice],
        mode="lines+markers", name=metric_choice,
        line=dict(color="#4472C4", width=2),
        yaxis="y2"
    ))

    fig.add_shape(
        type="line",
        x0=daily_chart["date"].min(), x1=daily_chart["date"].max(),
        y0=alert_val, y1=alert_val,
        line=dict(dash="dot", color="red"),
        yref="y2"
    )
    fig.add_annotation(
        x=daily_chart["date"].max(), y=alert_val,
        text=f"alert -{threshold}%",
        showarrow=False, yref="y2",
        xanchor="right", font=dict(color="red")
    )

    if metric_choice == "avg_not_free_credits":
        y1_format = ".2f"
    elif metric_choice in ["paid_spend_rate", "click_to_spend"]:
        y1_format = ".3%"
    else:
        y1_format = ".1%"

    fig.update_layout(
        height=350,
        margin=dict(t=20),
        yaxis=dict(title="Sends", showgrid=False),
        yaxis2=dict(tickformat=y1_format, title=metric_choice, overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        barmode="overlay"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Alert
    drops = daily_chart[daily_chart[metric_choice] < alert_val]["date"]
    if not drops.empty:
        st.error(f"⚠️ Critical drop below threshold on: {', '.join(drops.dt.strftime('%b %d').tolist())}")
    else:
        st.success("✅ No critical drops detected")

    # Heatmaps
    st.subheader("Rule × Response breakdown")
    hcol1, hcol2 = st.columns(2)

    pivot_sends = df_sub.pivot_table(
        index="rule", columns="response", values="is_read",
        aggfunc="count", fill_value=0
    )
    pivot_ctr = df_sub.pivot_table(
        index="rule", columns="response", values="is_clicked",
        aggfunc="mean", fill_value=0
    )

    with hcol1:
        fig_h1 = px.imshow(
            pivot_sends,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Sends volume",
            aspect="auto"
        )
        fig_h1.update_layout(height=300, margin=dict(t=40))
        st.plotly_chart(fig_h1, use_container_width=True)

    with hcol2:
        fig_h2 = px.imshow(
            pivot_ctr.round(3),
            text_auto=".1%",
            color_continuous_scale="RdYlGn",
            title="CTR",
            aspect="auto"
        )
        fig_h2.update_layout(height=300, margin=dict(t=40))
        st.plotly_chart(fig_h2, use_container_width=True)

    # AI Summary placeholder
    st.subheader("🤖 AI Summary")
    if st.button("Generate summary", key="mon_ai_btn"):
        if not groq_client:
            st.warning("Please add GROQ_API_KEY to .streamlit/secrets.toml")
        else:
            open_trend = "declining" if delta_open < 0 else "stable or improving"
            ctr_trend = "declining" if delta_ctr < 0 else "stable or improving"
            alert_dates = daily_chart[daily_chart[metric_choice] < alert_val]["date"].dt.strftime('%b %d').tolist()

            data_summary = f"""
                Period: {daily_chart['date'].min().strftime('%b %d')} to {daily_chart['date'].max().strftime('%b %d')}
                Segment: {segment} | Rule filter: {rule}
                Avg Open Rate: {avg_open:.2%}, Last Day Open: {last_open:.2%}, Trend: {open_trend}
                Avg CTR: {avg_ctr:.2%}, Last Day CTR: {last_ctr:.2%}, Trend: {ctr_trend}
                Selected Metric for Alerting: {metric_choice}
                Critical Drops Detected on Dates: {', '.join(alert_dates) if alert_dates else 'None'}
                """

            system_prompt = (
                "You are an expert Lead Product and Growth Analyst in a mobile tech company. Your job is to analyze email marketing metrics, "
                "identify root causes of anomalies, and write clear, concise executive summaries for the marketing team. "
                "Be brief, precise, focus on business logic (e.g., deliverability issues, technical bugs, bad targeting), "
                "and NEVER hallucinate stats not provided in the context."
            )

            user_prompt = f"""
                Based on the following aggregated metrics from our marketing dashboard, write an executive summary in English:
                {data_summary}

                Structure your response using markdown formatting:
                1. **Executive Summary** (Max 3 brief bullet points focusing on metrics dynamic).
                2. **Anomaly & Risk Analysis** (If alerts exist, suggest 2 realistic reasons for the drop on those specific days. If no alerts, comment on system health).
                3. **Action Items** (Provide 2 tactical next steps for the growth team).
                """

            with st.spinner("🤖 Groq is analyzing..."):
                try:
                    completion = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2,
                    )
                    st.markdown(completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Insights Generation Mistake: {e}")

# A/B ANALYSIS
else:
    st.title("🧪 A/B Test Analysis")

    group = st.selectbox("Test group", ["group_1", "group_2", "group_3", "group_4"])

    # Results for all segments
    segments = [("All", "All"), ("Buyer", "Buyer"), ("Not Buyer", "Not Buyer")]

    all_results = {}
    for segment, segment_label in segments:
        df_sub = df.copy()
        if segment != "All":
            df_sub = df_sub[df_sub["buyer"] == segment]

        rows = []
        for metric, label in [("is_read", "Open Rate"), ("is_clicked", "CTR"), ("is_paid_spend", "Paid Spend Rate")]:
            test = df_sub[df_sub[group] == "Test"]
            control = df_sub[df_sub[group] == "Control"]
            ct = pd.crosstab(df_sub[group], df_sub[metric])
            _, p, _, _ = chi2_contingency(ct) if ct.shape == (2, 2) else (None, 1.0, None, None)
            test_rate = test[metric].mean()
            control_rate = control[metric].mean()
            lift = test_rate / control_rate - 1 if control_rate else 0
            rows.append({
                "Metric": label,
                "Test": f"{test_rate:.3%}",
                "Control": f"{control_rate:.3%}",
                "Lift": f"{lift:+.1%}",
                "p-value": f"{p:.4f}",
                "Significant": "✅ Yes" if p < 0.05 else "❌ No",
                "_lift_val": lift,
                "_sig": p < 0.05,
            })
        all_results[segment_label] = rows

    # Three separate tables
    st.subheader(f"Results")
    for segment_label, rows in all_results.items():
        st.markdown(f"**{segment_label}**")
        rdf = pd.DataFrame(rows)

        def highlight_row(row):
            orig = rdf.iloc[row.name]
            if not orig["_sig"]:
                return [""] * len(row)
            color = "background-color: #d4edda" if orig["_lift_val"] > 0 else "background-color: #f8d7da"
            return [color] * len(row)

        st.dataframe(
            rdf.drop(columns=["_lift_val", "_sig"]).style.apply(highlight_row, axis=1),
            use_container_width=True, hide_index=True
        )
        st.divider()

    # Lift chart
    st.subheader("Lift by metric and segment")
    chart_rows = []
    for segment_label, rows in all_results.items():
        for r in rows:
            chart_rows.append({
                "Segment": segment_label,
                "Metric": r["Metric"],
                "Lift": r["_lift_val"],
                "Label": r["Lift"],
            })
    chart_df = pd.DataFrame(chart_rows)

    fig2 = px.bar(
        chart_df,
        x="Metric", y="Lift",
        color="Segment",
        barmode="group",
        color_discrete_map={"All": "#A9C4E8", "Buyer": "#4472C4", "Not Buyer": "#6c757d"},
        text="Label",
    )
    fig2.add_hline(y=0, line_color="black", line_width=1)
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig2, use_container_width=True)

    # AI recommendation
    st.subheader("🤖 AI Recommendation")
    if st.button("Generate recommendation", key="ab_ai_btn"):
        if not groq_client:
            st.warning("Please add GROQ_API_KEY to .streamlit/secrets.toml")
        else:
            test_results_context = ""
            for seg, rows in all_results.items():
                test_results_context += f"\nSegment: {seg}\n"
                for r in rows:
                    test_results_context += (
                        f"- Metric: {r['Metric']}, Test Rate: {r['Test']}, Control Rate: {r['Control']}, "
                        f"Lift: {r['Lift']}, p-value: {r['p-value']}, Stat Significant: {r['Significant']}\n"
                    )

            system_prompt = (
                "You are a Senior Data Scientist and Product Experimentation Expert. "
                "Your task is to interpret statistical A/B test results. "
                "You enforce strict statistical guardrails: if p-value >= 0.05, the result is NOT statistically significant, "
                "and you must strictly warn against implementing changes based on statistical noise, even if Lift looks positive."
            )

            user_prompt = f"""
                Analyze the following statistical test results for the experiment group '{group}':
                {test_results_context}

                Provide a professional analysis in English structured as follows:
                1. **Core Verdict**: Clear data-driven recommendation (Launch Test completely / Drop Test / Roll out only to specific segments / Continue testing).
                2. **Statistical Breakdown**: Briefly interpret why this choice is made based on p-values and segment-specific differences (especially focus on 'Buyer' vs 'Not Buyer' metrics).
                3. **Product Hypotheses**: Give a product-driven explanation (1-2 sentences) of why the feature might have performed this way.
                """

            with st.spinner("🤖 Groq is working!..."):
                try:
                    completion = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                    )
                    st.markdown(completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Recommendations Generation Mistake: {e}")