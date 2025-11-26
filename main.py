import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Exercise 1 - Serverless Cost Analysis",
    layout="wide"
)

st.title("Exercise 1: Identify Top Serverless Cost Contributors")

st.markdown(
    """
This dashboard focuses on:

1. **Finding which functions contribute ~80% of total serverless cost**  
2. **Visualizing Cost vs Invocation Frequency**
"""
)

# -------------------------------
# Load dataset from local file ONLY
# -------------------------------
CSV_PATH = "Serverless_Data.csv"

try:
    # quoting=3 == csv.QUOTE_NONE, so quotes are treated as normal chars
    df = pd.read_csv(CSV_PATH, sep=",", quoting=3)
except FileNotFoundError:
    st.error(f"❌ Could not find `{CSV_PATH}` in the current folder.")
    st.stop()

# Fix header names (strip extra quotes/spaces)
df.columns = df.columns.astype(str).str.replace('"', "").str.strip()

# Remove quotes from string cells and trim spaces
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.replace('"', "").str.strip()

# Coerce numeric columns to proper dtypes
numeric_cols = [
    "InvocationsPerMonth",
    "AvgDurationMs",
    "MemoryMB",
    "ColdStartRate",
    "ProvisionedConcurrency",
    "GBSeconds",
    "DataTransferGB",
    "CostUSD",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -------------------------------
# Quick debug – you can delete this once it's working
# -------------------------------
st.write("🔍 Columns:", list(df.columns))

# Basic validation
required_cols = [
    "FunctionName",
    "Environment",
    "InvocationsPerMonth",
    "AvgDurationMs",
    "MemoryMB",
    "ColdStartRate",
    "ProvisionedConcurrency",
    "GBSeconds",
    "DataTransferGB",
    "CostUSD",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.stop()

# -------------------------------
# Overview
# -------------------------------
st.subheader("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Functions", len(df))

with col2:
    total_cost = df["CostUSD"].sum()
    st.metric("Total Monthly Cost (USD)", f"${total_cost:,.2f}")

with col3:
    prod_cost = df[df["Environment"] == "production"]["CostUSD"].sum()
    st.metric("Production Cost (USD)", f"${prod_cost:,.2f}")

with st.expander("Show sample data"):
    st.dataframe(df.head())

# -------------------------------
# Exercise 1 – Top cost contributors
# -------------------------------
st.subheader("🎯 Top Cost Contributors (80% of Total Spend)")

df_cost = df.sort_values(by="CostUSD", ascending=False).reset_index(drop=True)

df_cost["CumulativeCost"] = df_cost["CostUSD"].cumsum()
df_cost["CumulativeCostPct"] = df_cost["CumulativeCost"] / total_cost * 100

threshold = 80
df_cost["Top80Flag"] = df_cost["CumulativeCostPct"] <= threshold

top80_df = df_cost[df_cost["Top80Flag"]]

num_top80 = len(top80_df)
pct_functions = num_top80 / len(df) * 100

st.markdown(
    f"""
**Result:**  
- **{num_top80} functions** (~{pct_functions:.1f}% of all functions)  
- Together account for **≈ {threshold}% of total serverless cost**.
"""
)

st.write("Table of functions contributing to ~80% of cost:")
st.dataframe(
    top80_df[
        [
            "FunctionName",
            "Environment",
            "InvocationsPerMonth",
            "CostUSD",
            "CumulativeCostPct",
        ]
    ]
)

# Cost vs Invocation Frequency
st.subheader("📈 Cost vs Invocation Frequency")

st.markdown(
    """
This scatter plot helps you see:
- **High-invocation, low-cost** functions (efficient)
- **Low-invocation, high-cost** functions (potential optimization targets)
"""
)

fig_scatter = px.scatter(
    df,
    x="InvocationsPerMonth",
    y="CostUSD",
    color="Environment",
    hover_name="FunctionName",
    size="CostUSD",
    size_max=20,
    title="CostUSD vs InvocationsPerMonth",
    labels={
        "InvocationsPerMonth": "Invocations per Month",
        "CostUSD": "Monthly Cost (USD)",
    },
)

st.plotly_chart(fig_scatter, use_container_width=True)

# -------------------------------
# Exercise 2 – Memory Right-Sizing
# -------------------------------
st.subheader("🧠 Exercise 2: Memory Right-Sizing")

st.markdown(
    """
Goal:

- Find functions where **memory is high** but **execution time is relatively low**  
- Estimate the **cost impact** of reducing memory (assuming duration stays about the same)

For simplicity, we assume Lambda cost is roughly proportional to  
**`MemoryMB × AvgDurationMs × InvocationsPerMonth`**,  
so lowering memory scales cost linearly.
"""
)

# Controls for what counts as "high memory" and "low duration"
col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    min_high_memory = st.slider(
        "Minimum memory (MB) to consider 'high'",
        min_value=256,
        max_value=4096,
        value=1024,
        step=128,
    )
with col_cfg2:
    max_low_duration = st.slider(
        "Maximum AvgDurationMs to consider 'low' (ms)",
        min_value=10,
        max_value=3000,
        value=300,
        step=10,
    )

# Filter candidate functions
candidates = df[
    (df["MemoryMB"] >= min_high_memory) &
    (df["AvgDurationMs"] <= max_low_duration)
].copy()

if candidates.empty:
    st.info(
        "No functions found that match the current thresholds for high memory "
        "and low duration. Try lowering the memory threshold or increasing the duration limit."
    )
else:
    # Suggest new memory = half the current, but not below 128 MB
    candidates["NewMemoryMB"] = (candidates["MemoryMB"] / 2).clip(lower=128)

    # Estimate new cost assuming linear scaling with memory
    candidates["EstimatedNewCostUSD"] = (
        candidates["CostUSD"] * (candidates["NewMemoryMB"] / candidates["MemoryMB"])
    )

    candidates["EstimatedSavingsUSD"] = (
        candidates["CostUSD"] - candidates["EstimatedNewCostUSD"]
    )
    candidates["EstimatedSavingsPct"] = (
        candidates["EstimatedSavingsUSD"] / candidates["CostUSD"] * 100
    )

    total_current = candidates["CostUSD"].sum()
    total_new = candidates["EstimatedNewCostUSD"].sum()
    total_savings = total_current - total_new

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Current Monthly Cost (Candidates)",
            f"${total_current:,.2f}"
        )
    with c2:
        st.metric(
            "Estimated New Cost After Right-Sizing",
            f"${total_new:,.2f}"
        )
    with c3:
        st.metric(
            "Estimated Savings",
            f"${total_savings:,.2f}",
            f"{(total_savings / total_current * 100):.1f}%"
        )

    st.write("📋 **Candidate functions for memory right-sizing:**")
    st.dataframe(
        candidates[
            [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "AvgDurationMs",
                "MemoryMB",
                "NewMemoryMB",
                "CostUSD",
                "EstimatedNewCostUSD",
                "EstimatedSavingsUSD",
                "EstimatedSavingsPct",
            ]
        ]
    )

    # Optional: visualize current vs new cost
    chart_df = candidates.melt(
        id_vars=["FunctionName"],
        value_vars=["CostUSD", "EstimatedNewCostUSD"],
        var_name="CostType",
        value_name="MonthlyCostUSD",
    )
    chart_df["CostType"] = chart_df["CostType"].replace({
        "CostUSD": "Current Cost",
        "EstimatedNewCostUSD": "Estimated New Cost",
    })

    fig_mem = px.bar(
        chart_df,
        x="FunctionName",
        y="MonthlyCostUSD",
        color="CostType",
        barmode="group",
        title="Current vs Estimated Cost After Memory Right-Sizing",
        labels={
            "FunctionName": "Function",
            "MonthlyCostUSD": "Monthly Cost (USD)",
        },
    )
    fig_mem.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_mem, use_container_width=True)

# -------------------------------
# Exercise 3 – Provisioned Concurrency Optimization
# -------------------------------
st.subheader("⚙️ Exercise 3: Provisioned Concurrency Optimization")

st.markdown(
    """
Goal:

- See how **Provisioned Concurrency (PC)** relates to **cold starts** and **cost**  
- Identify:
  - Functions where **PC is enabled but probably not needed** (low cold start rate, non-trivial PC units)  
  - Functions where **PC is disabled but cold starts are high** (potential candidates to enable PC)

Assumptions:
- We use **`ProvisionedConcurrency`** and **`ColdStartRate`** from the dataset.  
- We treat PC cost qualitatively using **total `CostUSD`** as a proxy.
"""
)

# Sliders for thresholds
c3_1, c3_2, c3_3 = st.columns(3)
with c3_1:
    min_pc_units = st.slider(
        "Min Provisioned Concurrency units to consider 'significant'",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
    )

with c3_2:
    max_cold_for_remove_pc_pct = st.slider(
        "Max cold start rate (%) to consider PC overkill (for remove/reduce)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
    )

with c3_3:
    min_cold_for_consider_pc_pct = st.slider(
        "Min cold start rate (%) to consider enabling PC",
        min_value=1.0,
        max_value=50.0,
        value=5.0,
        step=1.0,
    )

max_cold_for_remove_pc = max_cold_for_remove_pc_pct / 100.0
min_cold_for_consider_pc = min_cold_for_consider_pc_pct / 100.0

# Split into groups
with_pc = df[df["ProvisionedConcurrency"] > 0].copy()
without_pc = df[df["ProvisionedConcurrency"] == 0].copy()

# Candidates where PC might be reduced/removed:
# - PC > min_pc_units
# - ColdStartRate is already very low
over_provisioned_pc = with_pc[
    (with_pc["ProvisionedConcurrency"] >= min_pc_units) &
    (with_pc["ColdStartRate"] <= max_cold_for_remove_pc)
].copy()

# Candidates where PC might be added:
# - PC == 0
# - ColdStartRate is high
consider_enabling_pc = without_pc[
    without_pc["ColdStartRate"] >= min_cold_for_consider_pc
].copy()

# Summary metrics
col_pc1, col_pc2 = st.columns(2)
with col_pc1:
    st.markdown("### 🔻 Candidates to **reduce or remove** Provisioned Concurrency")
    st.write(
        f"Found **{len(over_provisioned_pc)}** function(s) where PC is enabled, "
        f"cold starts are already low (≤ {max_cold_for_remove_pc_pct:.1f}%), "
        "and PC units are significant."
    )
    if not over_provisioned_pc.empty:
        st.dataframe(
            over_provisioned_pc[
                [
                    "FunctionName",
                    "Environment",
                    "InvocationsPerMonth",
                    "ColdStartRate",
                    "ProvisionedConcurrency",
                    "CostUSD",
                ]
            ]
        )
    else:
        st.info("No over-provisioned PC candidates under the current thresholds.")

with col_pc2:
    st.markdown("### 🔺 Candidates to **consider enabling** Provisioned Concurrency")
    st.write(
        f"Found **{len(consider_enabling_pc)}** function(s) with **no PC** and "
        f"high cold start rate (≥ {min_cold_for_consider_pc_pct:.1f}%)."
    )
    if not consider_enabling_pc.empty:
        st.dataframe(
            consider_enabling_pc[
                [
                    "FunctionName",
                    "Environment",
                    "InvocationsPerMonth",
                    "ColdStartRate",
                    "ProvisionedConcurrency",
                    "CostUSD",
                ]
            ]
        )
    else:
        st.info("No candidates to enable PC under the current thresholds.")

# Visualization: ColdStartRate vs ProvisionedConcurrency
st.markdown("### 📊 Cold Start Rate vs Provisioned Concurrency (All Functions)")

fig_pc = px.scatter(
    df,
    x="ProvisionedConcurrency",
    y="ColdStartRate",
    size="CostUSD",
    color="Environment",
    hover_name="FunctionName",
    title="Cold Start Rate vs Provisioned Concurrency (bubble size = CostUSD)",
    labels={
        "ProvisionedConcurrency": "Provisioned Concurrency Units",
        "ColdStartRate": "Cold Start Rate (0–1)",
        "CostUSD": "Monthly Cost (USD)",
    },
)

st.plotly_chart(fig_pc, use_container_width=True)

# -------------------------------
# Exercise 4 – Detect Unused or Low-Value Workloads
# -------------------------------
st.subheader("🧹 Exercise 4: Detect Unused or Low-Value Workloads")

st.markdown(
    """
Goal:

- Find functions that:
  - Contribute **very little traffic** (low share of total invocations), but  
  - Still have **non-trivial or high cost**

This helps you identify **zombie / low-value functions** that might be:
- Old test utilities
- Forgotten dev/staging workloads
- Features no one really uses anymore
"""
)

total_invocations = df["InvocationsPerMonth"].sum()
total_cost = df["CostUSD"].sum()

df["InvocationSharePct"] = df["InvocationsPerMonth"] / total_invocations * 100
df["CostSharePct"] = df["CostUSD"] / total_cost * 100

c4_1, c4_2, c4_3 = st.columns(3)
with c4_1:
    max_invocation_share = st.slider(
        "Max % of total invocations to consider 'low traffic'",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

with c4_2:
    min_cost_share = st.slider(
        "Min % of total cost to consider 'non-trivial cost'",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
    )

with c4_3:
    min_absolute_cost = st.number_input(
        "Min absolute monthly cost (USD) to flag",
        min_value=0.0,
        value=5.0,
        step=1.0,
    )

low_value = df[
    (df["InvocationSharePct"] <= max_invocation_share) &
    (df["CostSharePct"] >= min_cost_share) &
    (df["CostUSD"] >= min_absolute_cost)
].copy()

if low_value.empty:
    st.info(
        "No functions match the current thresholds for 'low traffic but high cost'. "
        "Try increasing the max invocation % or lowering the cost thresholds."
    )
else:
    low_value = low_value.sort_values(by="CostUSD", ascending=False)

    total_low_value_cost = low_value["CostUSD"].sum()
    share_of_total_cost = total_low_value_cost / total_cost * 100

    c4m1, c4m2 = st.columns(2)
    with c4m1:
        st.metric(
            "Monthly Cost of Low-Value Functions",
            f"${total_low_value_cost:,.2f}"
        )
    with c4m2:
        st.metric(
            "Share of Total Cost",
            f"{share_of_total_cost:.2f}%"
        )

    st.write("📋 **Low-Value / Unused Candidates (low traffic, non-trivial cost):**")
    st.dataframe(
        low_value[
            [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "InvocationSharePct",
                "CostUSD",
                "CostSharePct",
            ]
        ]
    )

    # Optional: visualize them
    st.markdown("### 📊 Cost vs Invocation Share for Low-Value Candidates")
    fig_lv = px.scatter(
        low_value,
        x="InvocationSharePct",
        y="CostUSD",
        color="Environment",
        hover_name="FunctionName",
        size="CostUSD",
        size_max=20,
        title="Low-Value Functions: CostUSD vs Invocation Share (%)",
        labels={
            "InvocationSharePct": "Share of Total Invocations (%)",
            "CostUSD": "Monthly Cost (USD)",
        },
    )
    st.plotly_chart(fig_lv, use_container_width=True)

# -------------------------------
# Exercise 5 – Cost Forecasting Model
# -------------------------------
st.subheader("📐 Exercise 5: Cost Forecasting Model")

st.markdown(
    """
Goal:

- Build a simple model to approximate **CostUSD** as a function of:
  - **InvocationsPerMonth**
  - **AvgDurationMs**
  - **MemoryMB**
  - **DataTransferGB**

We use a linear model of the form:
""")

st.latex(r"""
CostUSD \approx 
\alpha \cdot (Invocations \times AvgDurationMs \times MemoryMB)
+ \beta \cdot DataTransferGB
+ \gamma
""")

# Compute a "compute work" feature
compute_work = (
    df["InvocationsPerMonth"]
    * df["AvgDurationMs"]
    * df["MemoryMB"]
)

data_transfer = df["DataTransferGB"]
y = df["CostUSD"].values

# Design matrix: [compute_work, data_transfer, 1]
X = np.column_stack(
    [
        compute_work.values,
        data_transfer.values,
        np.ones(len(df)),
    ]
)

# Solve least squares: minimize ||Xw - y||^2 -> w = [alpha, beta, gamma]
w, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
alpha, beta, gamma = w

# Predictions
y_pred = X @ w

# R^2
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

col5_1, col5_2, col5_3 = st.columns(3)
with col5_1:
    st.metric("α (Compute Coefficient)", f"{alpha:.6e}")
with col5_2:
    st.metric("β (Data Transfer Coefficient)", f"{beta:.6e}")
with col5_3:
    st.metric("γ (Intercept)", f"{gamma:.4f}")

st.markdown(f"**Model R² (goodness of fit):** `{r2:.4f}`")

# Show a small table of actual vs predicted
model_df = df.copy()
model_df["PredictedCostUSD"] = y_pred
model_df["ErrorUSD"] = model_df["CostUSD"] - model_df["PredictedCostUSD"]

st.write("📋 **Sample of Actual vs Predicted Cost:**")
st.dataframe(
    model_df[
        [
            "FunctionName",
            "Environment",
            "CostUSD",
            "PredictedCostUSD",
            "ErrorUSD",
        ]
    ].head(20)
)

# Plot: Actual vs Predicted
st.markdown("### 📊 Actual vs Predicted Monthly Cost")

fig_model = px.scatter(
    model_df,
    x="CostUSD",
    y="PredictedCostUSD",
    color="Environment",
    hover_name="FunctionName",
    title="Actual vs Predicted CostUSD",
    labels={
        "CostUSD": "Actual CostUSD",
        "PredictedCostUSD": "Predicted CostUSD",
    },
)
fig_model.add_shape(
    type="line",
    x0=model_df["CostUSD"].min(),
    y0=model_df["CostUSD"].min(),
    x1=model_df["CostUSD"].max(),
    y1=model_df["CostUSD"].max(),
    line=dict(dash="dash"),
)

st.plotly_chart(fig_model, use_container_width=True)

st.markdown(
    """
The closer the points are to the **dashed diagonal line**, the better the model
is at predicting the actual cost.

You can now re-use this fitted model to **forecast cost** if invocations,
duration, memory, or data transfer change.
"""
)

# -------------------------------
# Exercise 6 – Containerization Candidates
# -------------------------------
st.subheader("🐳 Exercise 6: Workloads That Might Benefit from Containerization")

st.markdown(
    """
Goal:

- Identify Lambda functions that might be more cost-effective or operationally
  better as **containers** (e.g., ECS/Fargate, Kubernetes):

Typical signals:

- **Long-running**: average duration > 3 seconds  
- **High memory**: ≥ 2 GB  
- **Low invocation frequency**: not called very often
"""
)

# Pre-computed helpers
df["AvgDurationSec"] = df["AvgDurationMs"] / 1000.0
df["InvocationsPerDay"] = df["InvocationsPerMonth"] / 30.0

# Slider bounds from data
min_dur = float(df["AvgDurationSec"].min())
max_dur = float(df["AvgDurationSec"].max())
min_mem = int(df["MemoryMB"].min())
max_mem = int(df["MemoryMB"].max())
min_inv = float(df["InvocationsPerMonth"].min())
max_inv = float(df["InvocationsPerMonth"].max())

c6_1, c6_2, c6_3 = st.columns(3)

with c6_1:
    min_duration_sec = st.slider(
        "Min Avg Duration (seconds) to consider 'long-running'",
        min_value=0.0,
        max_value=max(5.0, round(max_dur, 0)),
        value=3.0,
        step=0.5,
    )

with c6_2:
    min_memory_mb = st.slider(
        "Min Memory (MB) to consider 'high memory'",
        min_value=min_mem,
        max_value=max_mem,
        value=2048 if max_mem >= 2048 else max_mem,
        step=128,
    )

with c6_3:
    max_invocations_month = st.slider(
        "Max Invocations per Month to consider 'low frequency'",
        min_value=float(0),
        max_value=float(max_inv),
        value=float(max_inv / 4),
        step=float(max_inv / 20) if max_inv > 0 else 1.0,
    )

# Filter candidates
container_candidates = df[
    (df["AvgDurationSec"] >= min_duration_sec)
    & (df["MemoryMB"] >= min_memory_mb)
    & (df["InvocationsPerMonth"] <= max_invocations_month)
].copy()

if container_candidates.empty:
    st.info(
        "No functions match the current thresholds for long-running, high-memory, "
        "low-frequency workloads. Try lowering the duration/memory thresholds or "
        "raising the max invocations."
    )
else:
    container_candidates = container_candidates.sort_values(
        by="CostUSD", ascending=False
    )

    total_cand_cost = container_candidates["CostUSD"].sum()
    share_cand_cost = total_cand_cost / total_cost * 100

    c6m1, c6m2 = st.columns(2)
    with c6m1:
        st.metric(
            "Monthly Cost of Containerization Candidates",
            f"${total_cand_cost:,.2f}",
        )
    with c6m2:
        st.metric(
            "Share of Total Serverless Cost",
            f"{share_cand_cost:.2f}%",
        )

    st.write("📋 **Functions that may be better suited for containers:**")
    st.dataframe(
        container_candidates[
            [
                "FunctionName",
                "Environment",
                "InvocationsPerMonth",
                "InvocationsPerDay",
                "AvgDurationSec",
                "MemoryMB",
                "CostUSD",
            ]
        ]
    )

    st.markdown("### 📊 Cost of Containerization Candidates")
    fig_cont = px.bar(
        container_candidates,
        x="FunctionName",
        y="CostUSD",
        color="Environment",
        title="Monthly Cost of Potential Containerization Candidates",
        labels={
            "FunctionName": "Function",
            "CostUSD": "Monthly Cost (USD)",
        },
    )
    fig_cont.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_cont, use_container_width=True)
