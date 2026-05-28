import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io

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

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Filters")
tab_choice = st.sidebar.radio("Section", ["📈 Monitoring", "🧪 A/B Analysis"])

# ── MONITORING ───────────────────────────────────────────────────────────────

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

    # KPI cards
    st.subheader("KPIs")
    k1, k2, k3, k4 = st.columns(4)
    avg_open = daily["open_rate"].mean()
    avg_ctr  = daily["ctr"].mean()
    last_open = daily["open_rate"].iloc[-1]
    last_ctr  = daily["ctr"].iloc[-1]
    delta_open = last_open - avg_open
    delta_ctr  = last_ctr  - avg_ctr

    k1.metric("Avg Open Rate",  f"{avg_open:.1%}")
    k2.metric("Avg CTR",        f"{avg_ctr:.2%}")
    k3.metric("Last Day Open",  f"{last_open:.1%}", f"{delta_open:+.1%}")
    k4.metric("Last Day CTR",   f"{last_ctr:.2%}",  f"{delta_ctr:+.2%}")

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
    fig.add_trace(go.Scatter(
        x=daily_chart["date"], y=daily_chart[metric_choice],
        mode="lines+markers", name=metric_choice, line=dict(color="#4472C4", width=2)
    ))
    fig.add_hline(y=alert_val,  line_dash="dot",  line_color="red",   annotation_text=f"alert -{threshold}%")
    if metric_choice == "avg_not_free_credits":
        fig.update_yaxes(tickformat=".2f")
    elif metric_choice in ["paid_spend_rate", "click_to_spend"]:
        fig.update_yaxes(tickformat=".3%")
    else:
        fig.update_yaxes(tickformat=".1%")
    fig.update_layout(height=350, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    # Alert
    drops = daily[daily[metric_choice] < alert_val]["date"]
    if not drops.empty:
        st.error(f"⚠️ Critical drop below threshold on: {', '.join(drops.dt.strftime('%b %d').tolist())}")
    else:
        st.success("✅ No critical drops detected")

    # Rule breakdown
    st.subheader("Breakdown by Rule")
    rule_df = df_sub.groupby("rule").agg(
        sends=("is_read", "count"),
        open_rate=("is_read", "mean"),
        ctr=("is_clicked", "mean"),
    ).reset_index().sort_values("open_rate", ascending=False)
    rule_df["open_rate"] = rule_df["open_rate"].map("{:.1%}".format)
    rule_df["ctr"]       = rule_df["ctr"].map("{:.2%}".format)
    st.dataframe(rule_df, use_container_width=True, hide_index=True)

    # AI Summary placeholder
    st.subheader("🤖 AI Summary")
    if st.button("Generate summary"):
        open_trend = "declining" if delta_open < 0 else "stable or improving"
        ctr_trend  = "declining" if delta_ctr  < 0 else "stable or improving"
        n_alerts   = len(drops)
        summary = f"""
**Period:** {daily['date'].min().strftime('%b %d')} – {daily['date'].max().strftime('%b %d')}
**Segment:** {segment} | **Rule:** {rule}

- Open rate is **{open_trend}** (avg {avg_open:.1%}, last day {last_open:.1%})
- CTR is **{ctr_trend}** (avg {avg_ctr:.2%}, last day {last_ctr:.2%})
- **{n_alerts} alert day(s)** detected below the {threshold}% drop threshold

> ℹ️ Connect Claude API to get AI-generated narrative summary.
        """
        st.markdown(summary)

# A/B ANALYSIS

else:
    st.title("🧪 A/B Test Analysis")

    col1, col2 = st.columns(2)
    with col1:
        group = st.selectbox("Test group", ["group_1", "group_2", "group_3", "group_4"])
    with col2:
        segment = st.selectbox("Segment", ["All", "Buyer", "Not Buyer"])

    df_sub = df.copy()
    if segment != "All":
        df_sub = df_sub[df_sub["buyer"] == segment]

    # Results table
    results = []
    for metric, label in [("is_read", "Open Rate"), ("is_clicked", "CTR"), ("is_paid_spend", "Paid Spend Rate")]:
        test    = df_sub[df_sub[group] == "Test"]
        control = df_sub[df_sub[group] == "Control"]
        ct = pd.crosstab(df_sub[group], df_sub[metric])
        if ct.shape == (2, 2):
            _, p, _, _ = chi2_contingency(ct)
        else:
            p = 1.0
        test_rate = test[metric].mean()
        control_rate = control[metric].mean()
        lift   = test_rate / control_rate - 1 if control_rate else 0
        results.append({
            "Metric":        label,
            "Test":          f"{test_rate:.3%}",
            "Control":       f"{control_rate:.3%}",
            "Lift":          f"{lift:+.1%}",
            "p-value":       f"{p:.4f}",
            "Significant":   "✅ Yes" if p < 0.05 else "❌ No",
            "_lift_val":     lift,
            "_sig":          p < 0.05,
        })

    results_df = pd.DataFrame(results)

    # Highlight significant rows
    st.subheader(f"Results — {group} | {segment}")

    def highlight(row):
        if not row["_sig"]:
            return [""] * len(row)
        color = "background-color: #d4edda" if row["_lift_val"] > 0 else "background-color: #f8d7da"
        return [color] * len(row)

    display_df = results_df.drop(columns=["_lift_val", "_sig"])
    st.dataframe(
        results_df.style.apply(highlight, axis=1),
        use_container_width=True, hide_index=True
    )

    # Lift bar chart
    st.subheader("Lift by metric")
    fig2 = px.bar(
        results_df,
        x="Metric", y="_lift_val",
        color="_sig",
        color_discrete_map={True: "#4472C4", False: "#A9C4E8"},
        labels={"_lift_val": "Lift", "_sig": "Significant"},
        text=results_df["Lift"],
    )
    fig2.add_hline(y=0, line_color="black", line_width=1)
    fig2.update_yaxes(tickformat=".0%")
    fig2.update_layout(height=300, margin=dict(t=20), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # AI recommendation placeholder
    st.subheader("🤖 AI Recommendation")
    if st.button("Generate recommendation"):
        sig_results = [r for r in results if r["_sig"]]
        if not sig_results:
            rec = f"**{group}** shows no statistically significant effects for the **{segment}** segment. Consider running the test longer or expanding the sample."
        else:
            pos = [r for r in sig_results if r["_lift_val"] > 0]
            neg = [r for r in sig_results if r["_lift_val"] < 0]
            rec = f"**{group} | {segment}**\n\n"
            if pos:
                rec += "**Positive effects:** " + ", ".join([f"{r['Metric']} {r['Lift']}" for r in pos]) + "\n\n"
            if neg:
                rec += "**Negative effects:** " + ", ".join([f"{r['Metric']} {r['Lift']}" for r in neg]) + "\n\n"
            if pos and not neg:
                rec += "✅ **Recommendation: ship Test version.**"
            elif neg and not pos:
                rec += "❌ **Recommendation: do not ship. Test hurts key metrics.**"
            else:
                rec += "⚠️ **Mixed results. Review trade-offs before decision.**"
            rec += "\n\n> ℹ️ Connect Claude API to get AI-generated narrative recommendation."
        st.markdown(rec)
