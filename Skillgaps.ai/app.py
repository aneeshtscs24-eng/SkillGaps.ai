import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_generation import generate_employees, generate_courses
from models import RecommendationPipeline

# ── 1. PAGE CONFIGURATION & THEME ───────────────────────────────────────
st.set_page_config(
    page_title="SkillGaps.AI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS Styling Injection
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 20px;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .status-high { color: #dc3545; font-weight: bold; }
    .status-med { color: #ffc107; font-weight: bold; }
    .status-low { color: #198754; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ── 2. DATA ORCHESTRATION CACHING LAYER ─────────────────────────────────
@st.cache_data(show_spinner="Running Machine Learning Alignment Suite...")
def run_pipeline_execution():
    """Generates synthetic data and executes the complete semantic fairness pipeline."""
    # Generate reference datasets
    employees_df = generate_employees(50)
    courses_df = generate_courses()
    
    # Initialize and execute ML Pipeline matching top_n and attributes from main.py
    pipeline = RecommendationPipeline(top_n=3, model_name="all-MiniLM-L6-v2", protected_attribute="gender")
    logs_df = pipeline.run(employees_df, courses_df)
    
    # Extract structural metrics out of the fairness layer component
    before_parity = pipeline.fairness_layer.parity_score_before
    after_parity = pipeline.fairness_layer.fit(employees_df, pipeline.embedding_engine.compute_similarity_matrix(
        pipeline.embedding_engine.encode_skills(employees_df["skills_current"].tolist()),
        pipeline.embedding_engine.encode_courses(courses_df["description"].tolist())
    )).compute_adjusted_parity(employees_df, pipeline.fairness_layer.adjust_scores(
        employees_df, pipeline.embedding_engine.compute_similarity_matrix(
            pipeline.embedding_engine.encode_skills(employees_df["skills_current"].tolist()),
            pipeline.embedding_engine.encode_courses(courses_df["description"].tolist())
        )
    )[0])
    
    return employees_df, courses_df, logs_df, before_parity, after_parity

# ── 3. INTERFACE BUILDER ───────────────────────────────────────────────
st.title("🚀 SkillGaps.AI")
st.subheader("Ethical Workforce Transition & Explainable Recommendation Engine")
st.markdown("---")

# Execution Context Selection Sidebar
st.sidebar.header("⚙️ Configuration Engine")
mode = st.sidebar.selectbox("Pipeline Mode", ["Dry-Run Evaluation Mode (CSV)", "Live Database Mode (MySQL)"])

if st.sidebar.button("⚡ Run Pipeline Analysis", type="primary"):
    # Clear cache to simulate a fresh deployment run
    st.cache_data.clear()
    st.success("Pipeline executed successfully!")

# Load the core dataset structures dynamically
emp_df, course_df, logs_df, p_before, p_after = run_pipeline_execution()

# ── PHASE A: EXECUTIVE SUMMARY METRICS ──────────────────────────────────
st.header("📊 Executive Analytics Overview")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f'<div class="metric-card"><h5>Workforce Target Size</h5><h2>{len(emp_df)}</h2><p style="color:gray;">Active Employees</p></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><h5>Training Repositories</h5><h2>{len(course_df)}</h2><p style="color:gray;">NPTEL Courses</p></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><h5>Initial Parity Score</h5><h2>{p_before:.3f}</h2><p style="color:gray;">Raw Historical Bias Scale</p></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card" style="border-left-color:#198754;"><h5>Adjusted Parity Score</h5><h2>{p_after:.3f} 🎉</h2><p style="color:gray;">Demographic Parity Optimized</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Visualizing Recommendation Distributions with Plotly
st.subheader("⚖️ Post-Adjustment Equity Distribution Rates")
top1_recs = logs_df[logs_df["rank"] == 1]

# Calculate alignment metric ratios per gender group matching Phase 5 logic
gender_counts = emp_df["gender"].value_counts()
rec_counts = top1_recs["gender"].value_counts()
rates_data = []
for g in gender_counts.index:
    total_g = gender_counts[g]
    rec_g = rec_counts.get(g, 0)
    rate = (rec_g / total_g) * 100
    rates_data.append({"Demographic Group": g, "Recommendation Rate (%)": round(rate, 2)})

chart_df = pd.DataFrame(rates_data)
fig = px.bar(chart_df, x="Demographic Group", y="Recommendation Rate (%)", 
             color="Demographic Group", text="Recommendation Rate (%)",
             color_discrete_sequence=px.colors.qualitative.Pastel)
fig.update_layout(yaxis_range=[0, 100], showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── PHASE B: INTERACTIVE EMPLOYEE TRANSITION MATRIX ─────────────────────
st.header("🔍 Employee Transition Profile Explorer")

# Order choices using risk profile metrics descending matching Phase 3 rules
sorted_emp = emp_df.sort_values(by="risk_score", ascending=False)
emp_options = {f"#{row['employee_id']:02d} | {row['full_name']} ({row['role']})": row['employee_id'] for _, row in sorted_emp.iterrows()}

selected_key = st.selectbox("Select Employee Profile to Audit (Ordered by Transition Priority/Risk):", list(emp_options.keys()))
selected_id = emp_options[selected_key]

# Isolate employee details rows
employee_profile = emp_df[emp_df["employee_id"] == selected_id].iloc[0]
employee_recs = logs_df[logs_df["employee_id"] == selected_id].sort_values("rank")

# Render individual demographic profiles info pane
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"**Department:** {employee_profile['department']}")
    st.markdown(f"**Current Skills Profile:** `{employee_profile['skills_current']}`")
with c2:
    st.markdown(f"**Gender Attribute:** {employee_profile['gender']}")
    st.markdown(f"**Age / Operational Experience:** {employee_profile['age']} yrs old / {employee_profile['years_exp']} yrs exp")
with c3:
    risk_val = employee_profile['risk_score']
    if risk_val >= 0.55:
        status_label = f'<span class="status-high">CRITICAL RISK ({risk_val:.3f})</span>'
    elif risk_val >= 0.35:
        status_label = f'<span class="status-med">MODERATE RISK ({risk_val:.3f})</span>'
    else:
        status_label = f'<span class="status-low">STABLE PROFILE ({risk_val:.3f})</span>'
    st.markdown(f"**Displacement Vector:** {status_label}", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader(f"💡 Customized Professional Upskilling Tracks for {employee_profile['full_name']}")

# Loop across top recommendations dynamically mapped into expander nodes
for _, rec in employee_recs.iterrows():
    with st.expander(f"🏅 Rank #{int(rec['rank'])}: {rec['course_title']}", expanded=(rec['rank'] == 1)):
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.metric("Raw Semantic Match", f"{rec['raw_similarity']:.4f}")
            delta_val = rec['fairness_delta']
            st.metric("Ethical Parity Adjustment", f"{rec['fairness_score']:.4f}", 
                      delta=f"{delta_val:+.4f}" if abs(delta_val) > 0.001 else "0.0000")
            
        with col_right:
            st.markdown("**Algorithmic Explainable AI (XAI) Alignment Breakdown:**")
            # Format raw reason block line items cleanly into markdown code quote cards
            st.info(rec['xai_reason'])

st.markdown("---")
st.caption("SkillGaps.AI Engine Framework Dashboard UI • Verified Pipeline System Execution.")