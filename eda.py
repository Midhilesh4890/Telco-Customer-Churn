"""
TELECOM CHURN ANALYSIS - EXPLORATORY DATA ANALYSIS
File: eda.py
Purpose: Complete EDA with both standalone and Streamlit versions
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import streamlit (for streamlit version)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class TelecomEDAAnalyzer:
    """Complete Telecom Churn EDA with Plotly visualizations"""

    def __init__(self, data_file=None):
        self.df = None
        self.business_metrics = {}
        self.charts_dir = "eda_charts"

        # Create charts directory
        Path(self.charts_dir).mkdir(exist_ok=True)

        if data_file:
            self.load_data(data_file)

    def load_data(self, data_file):
        """Load and prepare data"""
        try:
            if isinstance(data_file, str):
                self.df = pd.read_csv(data_file)
            else:  # For streamlit uploaded file
                self.df = pd.read_csv(data_file)

            # Convert churn to string for consistency
            self.df['Churn'] = self.df['Churn'].astype(str)

            # Calculate derived metrics
            self.df['Total_Charges'] = (
                self.df['Total day charge'] +
                self.df['Total eve charge'] +
                self.df['Total night charge'] +
                self.df['Total intl charge']
            )

            self.df['Total_Minutes'] = (
                self.df['Total day minutes'] +
                self.df['Total eve minutes'] +
                self.df['Total night minutes'] +
                self.df['Total intl minutes']
            )

            self.df['Total_Calls'] = (
                self.df['Total day calls'] +
                self.df['Total eve calls'] +
                self.df['Total night calls'] +
                self.df['Total intl calls']
            )

            # Calculate business metrics
            self.calculate_business_metrics()
            return True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def calculate_business_metrics(self):
        """Calculate comprehensive business metrics"""
        if self.df is None:
            return {}

        # Basic metrics
        total_customers = len(self.df)
        churned_customers = (self.df['Churn'] == 'True').sum()
        retained_customers = total_customers - churned_customers
        churn_rate = (churned_customers / total_customers) * 100

        # Financial metrics
        avg_monthly_revenue = self.df['Total_Charges'].mean()
        churned_revenue = self.df[self.df['Churn']
                                  == 'True']['Total_Charges'].sum()
        retained_revenue = self.df[self.df['Churn']
                                   == 'False']['Total_Charges'].sum()
        monthly_revenue_at_risk = churned_revenue
        annual_revenue_at_risk = monthly_revenue_at_risk * 12

        # Customer behavior metrics
        churned = self.df[self.df['Churn'] == 'True']
        retained = self.df[self.df['Churn'] == 'False']

        avg_account_length_churned = churned['Account length'].mean()
        avg_account_length_retained = retained['Account length'].mean()
        avg_service_calls_churned = churned['Customer service calls'].mean()
        avg_service_calls_retained = retained['Customer service calls'].mean()

        self.business_metrics = {
            'total_customers': total_customers,
            'churned_customers': churned_customers,
            'retained_customers': retained_customers,
            'churn_rate': churn_rate,
            'avg_monthly_revenue': avg_monthly_revenue,
            'churned_avg_revenue': churned['Total_Charges'].mean(),
            'retained_avg_revenue': retained['Total_Charges'].mean(),
            'monthly_revenue_at_risk': monthly_revenue_at_risk,
            'annual_revenue_at_risk': annual_revenue_at_risk,
            'avg_account_length_churned': avg_account_length_churned,
            'avg_account_length_retained': avg_account_length_retained,
            'avg_service_calls_churned': avg_service_calls_churned,
            'avg_service_calls_retained': avg_service_calls_retained
        }

        return self.business_metrics

    def get_data_quality_summary(self):
        """Get comprehensive data quality summary"""
        if self.df is None:
            return {}

        summary = {
            'shape': self.df.shape,
            'missing_values': dict(self.df.isnull().sum()),
            'data_types': dict(self.df.dtypes.astype(str)),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'duplicate_rows': self.df.duplicated().sum(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }

        return summary

    def create_churn_distribution_chart(self):
        """Create interactive churn distribution chart"""
        churn_counts = self.df['Churn'].value_counts()
        churn_labels = ['Retained', 'Churned']
        values = [churn_counts.get('False', 0), churn_counts.get('True', 0)]
        colors = ['#2ecc71', '#e74c3c']

        fig = go.Figure(data=[go.Pie(
            labels=churn_labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent+value',
            textfont_size=14,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig.update_layout(
            title={
                'text': "Customer Churn Distribution",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            font=dict(size=14),
            showlegend=True,
            height=500,
            width=600
        )

        return fig

    def create_service_calls_impact_chart(self):
        """Create service calls impact on churn chart"""
        service_churn = self.df.groupby('Customer service calls').agg({
            'Churn': lambda x: (x == 'True').mean() * 100,
            'State': 'count'
        }).reset_index()
        service_churn.columns = ['Service_Calls',
                                 'Churn_Rate', 'Customer_Count']

        # Create dual axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add churn rate bar
        fig.add_trace(
            go.Bar(
                x=service_churn['Service_Calls'],
                y=service_churn['Churn_Rate'],
                name="Churn Rate (%)",
                marker_color='#e74c3c',
                hovertemplate='Service Calls: %{x}<br>Churn Rate: %{y:.1f}%<extra></extra>'
            ),
            secondary_y=False,
        )

        # Add customer count line
        fig.add_trace(
            go.Scatter(
                x=service_churn['Service_Calls'],
                y=service_churn['Customer_Count'],
                mode='lines+markers',
                name="Customer Count",
                line=dict(color='#3498db', width=3),
                marker=dict(size=8),
                hovertemplate='Service Calls: %{x}<br>Customer Count: %{y}<extra></extra>'
            ),
            secondary_y=True,
        )

        # Update layout
        fig.update_xaxes(title_text="Number of Service Calls")
        fig.update_yaxes(title_text="Churn Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Customers", secondary_y=True)

        fig.update_layout(
            title={
                'text': "Impact of Service Calls on Churn Rate",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=500,
            hovermode='x unified'
        )

        return fig

    def create_plan_analysis_chart(self):
        """Create plan analysis charts"""
        # International Plan Analysis
        intl_churn = pd.crosstab(
            self.df['International plan'],
            self.df['Churn'],
            normalize='index'
        ) * 100

        # Voice Mail Plan Analysis
        vmail_churn = pd.crosstab(
            self.df['Voice mail plan'],
            self.df['Churn'],
            normalize='index'
        ) * 100

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('International Plan Impact',
                            'Voice Mail Plan Impact'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # International plan chart
        fig.add_trace(
            go.Bar(
                x=intl_churn.index,
                y=intl_churn['False'] if 'False' in intl_churn.columns else [
                    0, 0],
                name='Retained',
                marker_color='#2ecc71',
                hovertemplate='Plan: %{x}<br>Retained: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=intl_churn.index,
                y=intl_churn['True'] if 'True' in intl_churn.columns else [
                    0, 0],
                name='Churned',
                marker_color='#e74c3c',
                hovertemplate='Plan: %{x}<br>Churned: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Voice mail plan chart
        fig.add_trace(
            go.Bar(
                x=vmail_churn.index,
                y=vmail_churn['False'] if 'False' in vmail_churn.columns else [
                    0, 0],
                name='Retained',
                marker_color='#2ecc71',
                showlegend=False,
                hovertemplate='Plan: %{x}<br>Retained: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(
                x=vmail_churn.index,
                y=vmail_churn['True'] if 'True' in vmail_churn.columns else [
                    0, 0],
                name='Churned',
                marker_color='#e74c3c',
                showlegend=False,
                hovertemplate='Plan: %{x}<br>Churned: %{y:.1f}%<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title={
                'text': "Churn Rate by Service Plans",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=500,
            barmode='group'
        )

        fig.update_yaxes(title_text="Percentage (%)")

        return fig

    def create_account_length_distribution(self):
        """Create account length distribution chart"""
        retained = self.df[self.df['Churn'] == 'False']['Account length']
        churned = self.df[self.df['Churn'] == 'True']['Account length']

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=retained,
            name='Retained',
            opacity=0.7,
            marker_color='#2ecc71',
            nbinsx=30,
            hovertemplate='Account Length: %{x}<br>Count: %{y}<extra></extra>'
        ))

        fig.add_trace(go.Histogram(
            x=churned,
            name='Churned',
            opacity=0.7,
            marker_color='#e74c3c',
            nbinsx=30,
            hovertemplate='Account Length: %{x}<br>Count: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': "Account Length Distribution by Churn Status",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Account Length (days)",
            yaxis_title="Number of Customers",
            barmode='overlay',
            height=500
        )

        return fig

    def create_revenue_distribution(self):
        """Create revenue distribution chart"""
        retained = self.df[self.df['Churn'] == 'False']['Total_Charges']
        churned = self.df[self.df['Churn'] == 'True']['Total_Charges']

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=retained,
            name='Retained',
            opacity=0.7,
            marker_color='#2ecc71',
            nbinsx=30,
            hovertemplate='Monthly Charges: $%{x:.2f}<br>Count: %{y}<extra></extra>'
        ))

        fig.add_trace(go.Histogram(
            x=churned,
            name='Churned',
            opacity=0.7,
            marker_color='#e74c3c',
            nbinsx=30,
            hovertemplate='Monthly Charges: $%{x:.2f}<br>Count: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': "Monthly Revenue Distribution by Churn Status",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Total Monthly Charges ($)",
            yaxis_title="Number of Customers",
            barmode='overlay',
            height=500
        )

        return fig

    def create_usage_patterns_scatter(self):
        """Create usage patterns scatter plot"""
        retained = self.df[self.df['Churn'] == 'False']
        churned = self.df[self.df['Churn'] == 'True']

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=retained['Total day minutes'],
            y=retained['Total day charge'],
            mode='markers',
            name='Retained',
            marker=dict(color='#2ecc71', size=6, opacity=0.6),
            hovertemplate='Minutes: %{x:.1f}<br>Charge: $%{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=churned['Total day minutes'],
            y=churned['Total day charge'],
            mode='markers',
            name='Churned',
            marker=dict(color='#e74c3c', size=6, opacity=0.6),
            hovertemplate='Minutes: %{x:.1f}<br>Charge: $%{y:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': "Day Usage Patterns: Minutes vs Charges",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Total Day Minutes",
            yaxis_title="Total Day Charge ($)",
            height=500
        )

        return fig

    def create_state_analysis(self):
        """Create state-wise churn analysis"""
        state_analysis = self.df.groupby('State').agg({
            'Churn': lambda x: (x == 'True').mean() * 100,
            'Account length': 'count'
        }).reset_index()
        state_analysis.columns = ['State', 'Churn_Rate', 'Customer_Count']

        # Get top 15 states by churn rate
        top_states = state_analysis.nlargest(15, 'Churn_Rate')

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=top_states['State'],
            y=top_states['Churn_Rate'],
            marker_color='#f39c12',
            hovertemplate='State: %{x}<br>Churn Rate: %{y:.1f}%<br>Customers: %{customdata}<extra></extra>',
            customdata=top_states['Customer_Count']
        ))

        fig.update_layout(
            title={
                'text': "Top 15 States by Churn Rate",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="State",
            yaxis_title="Churn Rate (%)",
            height=500
        )

        return fig

    def create_correlation_heatmap(self):
        """Create correlation heatmap"""
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': "Feature Correlation Matrix",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=800,
            height=700
        )

        return fig

    def generate_all_charts(self):
        """Generate all EDA charts"""
        charts = {}

        print("Generating EDA charts...")

        charts['churn_distribution'] = self.create_churn_distribution_chart()
        print("Churn distribution chart created")

        charts['service_calls_impact'] = self.create_service_calls_impact_chart()
        print("Service calls impact chart created")

        charts['plan_analysis'] = self.create_plan_analysis_chart()
        print("Plan analysis chart created")

        charts['account_length'] = self.create_account_length_distribution()
        print("Account length distribution chart created")

        charts['revenue_distribution'] = self.create_revenue_distribution()
        print("Revenue distribution chart created")

        charts['usage_patterns'] = self.create_usage_patterns_scatter()
        print("Usage patterns scatter plot created")

        charts['state_analysis'] = self.create_state_analysis()
        print("State analysis chart created")

        charts['correlation_heatmap'] = self.create_correlation_heatmap()
        print("Correlation heatmap created")

        print(f"\nAll charts generated successfully")
        return charts

# Standalone version


def run_standalone_eda(data_file="churnbigml80.csv"):
    """Run standalone EDA analysis"""
    print("TELECOM CHURN - EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    # Initialize analyzer
    analyzer = TelecomEDAAnalyzer(data_file)

    if analyzer.df is None:
        print("Error: Could not load data file")
        return None, None

    print(f"Data loaded successfully: {analyzer.df.shape}")

    # Display business metrics
    print("\nBUSINESS METRICS:")
    print("-" * 20)
    metrics = analyzer.business_metrics
    print(f"Total Customers: {metrics['total_customers']:,}")
    print(f"Churn Rate: {metrics['churn_rate']:.1f}%")
    print(f"Avg Monthly Revenue: ${metrics['avg_monthly_revenue']:.2f}")
    print(f"Annual Revenue at Risk: ${metrics['annual_revenue_at_risk']:,.0f}")

    # Display data quality summary
    print("\nDATA QUALITY SUMMARY:")
    print("-" * 25)
    quality = analyzer.get_data_quality_summary()
    print(f"Dataset Shape: {quality['shape']}")
    print(f"Memory Usage: {quality['memory_usage']}")
    print(f"Missing Values: {sum(quality['missing_values'].values())}")
    print(f"Duplicate Rows: {quality['duplicate_rows']}")

    # Generate all charts
    print("\nGENERATING VISUALIZATIONS:")
    print("-" * 30)
    charts = analyzer.generate_all_charts()

    print(f"\nAnalysis complete! {len(charts)} interactive charts generated.")

    return analyzer, charts

# Streamlit version


def create_streamlit_app():
    """Create Streamlit EDA application"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available. Please install: pip install streamlit")
        return

    # Page configuration
    st.set_page_config(
        page_title="Telecom Churn Analysis - EDA",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title and description
    st.title("Telecom Churn Analysis - Interactive EDA")
    st.markdown(
        "Comprehensive exploratory data analysis for telecom customer churn prediction")

    # Sidebar for data upload and navigation
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your telecom churn dataset"
    )

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None

    # Load data
    if uploaded_file is not None:
        if st.session_state.analyzer is None:
            with st.spinner("Loading data..."):
                analyzer = TelecomEDAAnalyzer()
                if analyzer.load_data(uploaded_file):
                    st.session_state.analyzer = analyzer
                    st.sidebar.success("Data loaded successfully!")
                else:
                    st.sidebar.error("Error loading data")

    # Try to load default data if no file uploaded
    elif st.session_state.analyzer is None:
        try:
            analyzer = TelecomEDAAnalyzer("churnbigml80.csv")
            if analyzer.df is not None:
                st.session_state.analyzer = analyzer
                st.sidebar.info("Using default dataset")
        except:
            st.warning("Please upload a CSV file to begin analysis")
            return

    analyzer = st.session_state.analyzer

    if analyzer is None or analyzer.df is None:
        st.warning("Please upload a valid CSV file to begin analysis")
        return

    # Sidebar navigation
    st.sidebar.header("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis",
        [
            "Overview & Metrics",
            "Data Quality",
            "Churn Distribution",
            "Service Quality Impact",
            "Plan Analysis",
            "Customer Demographics",
            "Usage Patterns",
            "Geographic Analysis",
            "Feature Correlations"
        ]
    )

    # Main content area
    if analysis_type == "Overview & Metrics":
        st.header("Business Metrics Overview")

        # Display metrics cards
        col1, col2, col3, col4 = st.columns(4)

        metrics = analyzer.business_metrics

        with col1:
            st.metric(
                label="Total Customers",
                value=f"{metrics['total_customers']:,}",
                delta=None
            )

        with col2:
            st.metric(
                label="Churn Rate",
                value=f"{metrics['churn_rate']:.1f}%",
                delta="Critical" if metrics['churn_rate'] > 20 else "Good"
            )

        with col3:
            st.metric(
                label="Avg Monthly Revenue",
                value=f"${metrics['avg_monthly_revenue']:.2f}",
                delta=f"${metrics['retained_avg_revenue'] - metrics['churned_avg_revenue']:+.2f} vs churned"
            )

        with col4:
            st.metric(
                label="Annual Revenue at Risk",
                value=f"${metrics['annual_revenue_at_risk']:,.0f}",
                delta="High Risk"
            )

        # Additional metrics table
        st.subheader("Detailed Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Customer Metrics**")
            customer_metrics = pd.DataFrame({
                'Metric': [
                    'Total Customers',
                    'Churned Customers',
                    'Retained Customers',
                    'Churn Rate'
                ],
                'Value': [
                    f"{metrics['total_customers']:,}",
                    f"{metrics['churned_customers']:,}",
                    f"{metrics['retained_customers']:,}",
                    f"{metrics['churn_rate']:.1f}%"
                ]
            })
            st.dataframe(customer_metrics, use_container_width=True)

        with col2:
            st.markdown("**Financial Metrics**")
            financial_metrics = pd.DataFrame({
                'Metric': [
                    'Avg Monthly Revenue',
                    'Churned Customer Avg Revenue',
                    'Retained Customer Avg Revenue',
                    'Monthly Revenue at Risk',
                    'Annual Revenue at Risk'
                ],
                'Value': [
                    f"${metrics['avg_monthly_revenue']:.2f}",
                    f"${metrics['churned_avg_revenue']:.2f}",
                    f"${metrics['retained_avg_revenue']:.2f}",
                    f"${metrics['monthly_revenue_at_risk']:,.0f}",
                    f"${metrics['annual_revenue_at_risk']:,.0f}"
                ]
            })
            st.dataframe(financial_metrics, use_container_width=True)

    elif analysis_type == "Data Quality":
        st.header("Data Quality Assessment")

        quality = analyzer.get_data_quality_summary()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Overview")
            overview_df = pd.DataFrame({
                'Metric': ['Rows', 'Columns', 'Memory Usage', 'Duplicate Rows'],
                'Value': [
                    f"{quality['shape'][0]:,}",
                    f"{quality['shape'][1]}",
                    quality['memory_usage'],
                    f"{quality['duplicate_rows']:,}"
                ]
            })
            st.dataframe(overview_df, use_container_width=True)

        with col2:
            st.subheader("Missing Values")
            missing_df = pd.DataFrame.from_dict(
                quality['missing_values'],
                orient='index',
                columns=['Missing Count']
            ).reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df = missing_df[missing_df['Missing Count'] > 0]

            if missing_df.empty:
                st.success("No missing values found!")
            else:
                st.dataframe(missing_df, use_container_width=True)

        st.subheader("Data Types")
        dtypes_df = pd.DataFrame.from_dict(
            quality['data_types'],
            orient='index',
            columns=['Data Type']
        ).reset_index()
        dtypes_df.columns = ['Column', 'Data Type']
        st.dataframe(dtypes_df, use_container_width=True)

        # Sample data
        st.subheader("Sample Data")
        st.dataframe(analyzer.df.head(10), use_container_width=True)

    elif analysis_type == "Churn Distribution":
        st.header("Customer Churn Distribution")

        fig = analyzer.create_churn_distribution_chart()
        st.plotly_chart(fig, use_container_width=True)

        # Additional insights
        st.subheader("Key Insights")
        metrics = analyzer.business_metrics

        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Current Situation:**
            - {metrics['churn_rate']:.1f}% of customers are churning
            - This represents {metrics['churned_customers']:,} customers
            - Industry benchmark is typically 15-20%
            """)

        with col2:
            benchmark = "above average" if metrics['churn_rate'] > 20 else "at average" if metrics[
                'churn_rate'] > 15 else "below average"
            st.warning(f"""
            **Business Impact:**
            - Churn rate is {benchmark}
            - ${metrics['annual_revenue_at_risk']:,.0f} annual revenue at risk
            - Average churned customer value: ${metrics['churned_avg_revenue']:.2f}
            """)

    elif analysis_type == "Service Quality Impact":
        st.header("Service Quality Impact on Churn")

        fig = analyzer.create_service_calls_impact_chart()
        st.plotly_chart(fig, use_container_width=True)

        # Service calls analysis table
        st.subheader("Service Calls Analysis")

        service_analysis = analyzer.df.groupby('Customer service calls').agg({
            'Churn': lambda x: (x == 'True').mean() * 100,
            'State': 'count',
            'Total_Charges': 'mean'
        }).round(2)
        service_analysis.columns = [
            'Churn Rate (%)', 'Customer Count', 'Avg Revenue ($)']
        service_analysis = service_analysis.reset_index()

        st.dataframe(service_analysis, use_container_width=True)

    elif analysis_type == "Plan Analysis":
        st.header("Service Plan Analysis")

        fig = analyzer.create_plan_analysis_chart()
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Customer Demographics":
        st.header("Customer Demographics Analysis")

        # Account length distribution
        fig = analyzer.create_account_length_distribution()
        st.plotly_chart(fig, use_container_width=True)

        # Revenue distribution
        st.subheader("Revenue Distribution")
        fig2 = analyzer.create_revenue_distribution()
        st.plotly_chart(fig2, use_container_width=True)

    elif analysis_type == "Usage Patterns":
        st.header("Customer Usage Patterns")

        fig = analyzer.create_usage_patterns_scatter()
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Geographic Analysis":
        st.header("Geographic Analysis")

        fig = analyzer.create_state_analysis()
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Feature Correlations":
        st.header("Feature Correlation Analysis")

        fig = analyzer.create_correlation_heatmap()
        st.plotly_chart(fig, use_container_width=True)


# Main execution
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        create_streamlit_app()
    else:
        analyzer, charts = run_standalone_eda()

        if analyzer and charts:
            print("\nTo run the Streamlit version, use:")
            print("streamlit run eda.py streamlit")
