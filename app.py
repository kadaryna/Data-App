import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Email Marketing Dashboard", layout="wide")

# ── Synthetic data ──────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2024-10-02", "2024-10-10", freq="D")
    rules = ["FH", "FD", "FW", "SW"]
    rule_weights = [0.08, 0.59, 0.28, 0.05]

    df = pd.DataFrame({
        "date": np.random.choice(dates, n),
        "buyer": np.random.choice(["Buyer", "Non-Buyer"], n, p=[0.097, 0.903]),
        "rule": np.random.choice(rules, n, p=rule_weights),
        "group_1": np.random.choice(["Test", "Control"], n),
        "group_2": np.random.choice(["Test", "Control"], n),
        "group_3": np.random.choice(["Test", "Control"], n),
        "group_4": np.random.choice(["Test", "Control"], n),
    })

    # Realistic open/click rates per rule
    open_rate_map = {"FH": 0.39, "FD": 0.20, "FW": 0.05, "SW": 0.02}
    ctr_map       = {"FH": 0.17, "FD": 0.014, "FW": 0.005, "SW": 0.004}
    buyer_open_boost = {"Buyer": 1.35, "Non-Buyer": 1.0}
    buyer_ctr_boost  = {"Buyer": 2.0,  "Non-Buyer": 1.0}

    # group_1 Test boosts CTR for buyers
    ctr_group1_boost = np.where(
        (df["group_1"] == "Test") & (df["buyer"] == "Buyer"), 1.65, 1.0
    )

    df["is_read"] = np.random.binomial(
        1,
        np.clip(
            df["rule"].map(open_rate_map) * df["buyer"].map(buyer_open_boost), 0, 1
        )
    )
    df["is_clicked"] = np.random.binomial(
        1,
        np.clip(
            df["rule"].map(ctr_map)
            * df["buyer"].map(buyer_ctr_boost)
            * ctr_group1_boost, 0, 1
        )
    )
    df["is_paid_spend"] = np.random.binomial(
        1,
        np.where(df["buyer"] == "Buyer", 0.002, 0.0001)
    )

    return df

df = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Filters")
tab_choice = st.sidebar.radio("Section", ["📈 Monitoring", "🧪 A/B Analysis"])

# ── MONITORING ───────────────────────────────────────────────────────────────

if tab_choice == "📈 Monitoring":
    st.title("📈 Email Monitoring")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        segment = st.selectbox("Segment", ["All", "Buyer", "Non-Buyer"])
    with col2:
        rule = st.selectbox("Rule", ["All"] + sorted(df["rule"].unique().tolist()))
    with col3:
        threshold = st.slider("Alert threshold (drop %)", 5, 30, 15)

    # Apply filters
    dff = df.copy()
    if segment != "All":
        dff = dff[dff["buyer"] == segment]
    if rule != "All":
        dff = dff[dff["rule"] == rule]

    # Daily metrics
    daily = dff.groupby("date").agg(
        sends=("is_read", "count"),
        opens=("is_read", "sum"),
        clicks=("is_clicked", "sum"),
    ).reset_index()
    daily["open_rate"]  = daily["opens"]  / daily["sends"]
    daily["ctr"]        = daily["clicks"] / daily["sends"]
    daily["open_to_click"] = daily["clicks"] / daily["opens"].replace(0, np.nan)

    # KPI cards
    st.subheader("Overall KPIs")
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
    metric_choice = st.selectbox("Metric", ["open_rate", "ctr", "open_to_click"])

    mean_val = daily[metric_choice].mean()
    alert_val = mean_val * (1 - threshold / 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily[metric_choice],
        mode="lines+markers", name=metric_choice, line=dict(color="#4472C4", width=2)
    ))
    fig.add_hline(y=mean_val,   line_dash="dash", line_color="gray",  annotation_text="avg")
    fig.add_hline(y=alert_val,  line_dash="dot",  line_color="red",   annotation_text=f"alert -{threshold}%")
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
    rule_df = dff.groupby("rule").agg(
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

# ── A/B ANALYSIS ─────────────────────────────────────────────────────────────

else:
    st.title("🧪 A/B Test Analysis")

    col1, col2 = st.columns(2)
    with col1:
        group = st.selectbox("Test group", ["group_1", "group_2", "group_3", "group_4"])
    with col2:
        segment = st.selectbox("Segment", ["All", "Buyer", "Non-Buyer"])

    dff = df.copy()
    if segment != "All":
        dff = dff[dff["buyer"] == segment]

    # Results table
    results = []
    for metric, label in [("is_read", "Open Rate"), ("is_clicked", "CTR"), ("is_paid_spend", "Paid Spend Rate")]:
        test    = dff[dff[group] == "Test"]
        control = dff[dff[group] == "Control"]
        ct = pd.crosstab(dff[group], dff[metric])
        if ct.shape == (2, 2):
            _, p, _, _ = chi2_contingency(ct)
        else:
            p = 1.0
        t_rate = test[metric].mean()
        c_rate = control[metric].mean()
        lift   = t_rate / c_rate - 1 if c_rate else 0
        results.append({
            "Metric":        label,
            "Test":          f"{t_rate:.3%}",
            "Control":       f"{c_rate:.3%}",
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
