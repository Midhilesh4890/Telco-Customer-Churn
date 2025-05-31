# ------------------------------ Imports ------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ------------------------------ Page Configuration ------------------------------
st.set_page_config(
    page_title="Customer Churn - Business Recommendations",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------ Custom CSS ------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .roi-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .priority-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .priority-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-highlight {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------ Load Data ------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("/mnt/data/churnbigml80.csv")
        df.columns = df.columns.str.strip()
        df['Churn'] = df['Churn'].map({'True': True, 'False': False})

        df['Total_minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
        df['Total_calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
        df['Total_charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ------------------------------ Business Impact ------------------------------
def calculate_business_impact(df):
    total_customers = len(df)
    churned_customers = df['Churn'].sum()
    churn_rate = df['Churn'].mean() * 100
    avg_monthly_revenue = df['Total_charge'].mean()
    avg_annual_revenue = avg_monthly_revenue * 12
    annual_churn_loss = churned_customers * avg_annual_revenue
    customer_acquisition_cost = 150
    avg_account_length_months = df['Account length'].mean()
    customer_lifetime_value = avg_monthly_revenue * avg_account_length_months

    return {
        'total_customers': total_customers,
        'churned_customers': churned_customers,
        'churn_rate': churn_rate,
        'avg_monthly_revenue': avg_monthly_revenue,
        'avg_annual_revenue': avg_annual_revenue,
        'annual_churn_loss': annual_churn_loss,
        'customer_acquisition_cost': customer_acquisition_cost,
        'customer_lifetime_value': customer_lifetime_value
    }

# ------------------------------ Financial Impact (with fix) ------------------------------
def show_financial_impact(df, metrics):
    st.markdown('<h2 class="section-header">💰 Financial Impact Analysis</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="roi-card">
            <h3>Current Annual Loss</h3>
            <div class="metric-highlight">${metrics['annual_churn_loss']:,.0f}</div>
            <p>Revenue lost to churn</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="roi-card">
            <h3>Retention Opportunity</h3>
            <div class="metric-highlight">${metrics['annual_churn_loss'] * 0.3:,.0f}</div>
            <p>Potential revenue protection</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="roi-card">
            <h3>Implementation Investment</h3>
            <div class="metric-highlight">$800K</div>
            <p>Total program cost</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Return on Investment Analysis")
    investment = 800000
    potential_revenue_protection = metrics['annual_churn_loss'] * 0.3
    annual_roi = ((potential_revenue_protection - investment) / investment) * 100

    col1, col2 = st.columns(2)
    with col1:
        roi_data = {
            'Metric': [
                'Total Investment', 'Annual Revenue Protection',
                'Net Annual Benefit', 'ROI Percentage', 'Payback Period'
            ],
            'Value': [
                f'-${investment:,.0f}',
                f'${potential_revenue_protection:,.0f}',
                f'${potential_revenue_protection - investment:,.0f}',
                f'{annual_roi:.1f}%',
                f'{investment / potential_revenue_protection * 12:.1f} months'
            ]
        }
        roi_df = pd.DataFrame(roi_data)
        st.dataframe(roi_df, use_container_width=True)

    with col2:
        years = ['Year 1', 'Year 2', 'Year 3']
        cumulative_investment = [800, 800, 800]
        cumulative_savings = [855, 1710, 2565]
        net_benefit = [s - i for s, i in zip(cumulative_savings, cumulative_investment)]

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Investment', x=years, y=cumulative_investment, marker_color='red'))
        fig.add_trace(go.Bar(name='Revenue Protection', x=years, y=cumulative_savings, marker_color='green'))
        fig.add_trace(go.Scatter(name='Net Benefit', x=years, y=net_benefit, mode='lines+markers',
                                 line=dict(color='blue', width=3), yaxis='y2'))
        fig.update_layout(
            title='3-Year Financial Projection ($000s)',
            xaxis_title='Year',
            yaxis_title='Amount ($000s)',
            yaxis2=dict(title='Net Benefit ($000s)', overlaying='y', side='right'),
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Customer Segment Value Analysis")

    # Fixing the KeyError by ensuring all expected columns are present
    df['Revenue_Quartile'] = pd.qcut(df['Total_charge'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    segment_analysis = df.groupby(['Revenue_Quartile', 'Churn']).size().unstack(fill_value=0)
    if True not in segment_analysis.columns:
        segment_analysis[True] = 0
    if False not in segment_analysis.columns:
        segment_analysis[False] = 0

    segment_analysis['Churn_Rate'] = segment_analysis[True] / (segment_analysis[True] + segment_analysis[False]) * 100
    segment_analysis['Annual_Revenue_at_Risk'] = segment_analysis[True] * df.groupby('Revenue_Quartile')['Total_charge'].mean() * 12

    st.dataframe(segment_analysis[['Churn_Rate', 'Annual_Revenue_at_Risk']], use_container_width=True)

# ------------------------------ Main Entry ------------------------------
def main():
    st.markdown('<h1 class="main-header">💼 Customer Churn - Business Recommendations</h1>', unsafe_allow_html=True)

    df = load_data()
    if df is None:
        st.stop()

    metrics = calculate_business_impact(df)

    st.sidebar.title("📋 Navigation")
    section = st.sidebar.selectbox("Choose Section:", ["Financial Impact"])

    if section == "Financial Impact":
        show_financial_impact(df, metrics)

if __name__ == "__main__":
    main()
