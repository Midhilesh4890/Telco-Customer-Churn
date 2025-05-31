"""
TELECOM CHURN ANALYSIS - MODEL TRAINING (FIXED VERSION)
File: model_training_fixed.py
Purpose: Model training with interactive Plotly charts - Fixed chart generation issues
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class TelecomModelTrainer:
    """Model Training with Plotly visualizations for Telecom Churn Analysis"""

    def __init__(self, data_file='churnbigml80.csv'):  # Fixed path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}
        self.business_impact = {}
        self.feature_importance = None
        self.charts_dir = "model_charts"

        # Create charts directory
        Path(self.charts_dir).mkdir(exist_ok=True)

        if data_file and os.path.exists(data_file):
            self.load_and_prepare_data(data_file)
        else:
            print(
                f"Warning: Data file '{data_file}' not found. Please load data manually using load_and_prepare_data()")

    def load_and_prepare_data(self, data_file):
        """Load and prepare modeling data"""
        try:
            print(f"Loading data from: {data_file}")
            self.df = pd.read_csv(data_file)
            print(f"Data loaded successfully: {self.df.shape}")

            # Data preprocessing
            self.df = self.df.copy()

            # Handle Churn column conversion more robustly
            if 'Churn' not in self.df.columns:
                print("Error: 'Churn' column not found in data")
                return False

            # Convert boolean/string Churn to numeric
            if self.df['Churn'].dtype == 'bool':
                self.df['Churn'] = self.df['Churn'].astype(int)
            elif self.df['Churn'].dtype == 'object':
                # Handle string values like 'True'/'False' or 'Yes'/'No'
                churn_mapping = {'True': 1, 'False': 0,
                                 'Yes': 1, 'No': 0, 'true': 1, 'false': 0}
                if self.df['Churn'].iloc[0] in churn_mapping:
                    self.df['Churn'] = self.df['Churn'].map(churn_mapping)
                else:
                    # If already numeric strings, convert directly
                    self.df['Churn'] = pd.to_numeric(
                        self.df['Churn'], errors='coerce')

            # Encode categorical variables
            label_encoders = {}
            categorical_columns = [
                'State', 'International plan', 'Voice mail plan']

            for col in categorical_columns:
                if col in self.df.columns:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    label_encoders[col] = le

            # Create additional features if the required columns exist
            charge_cols = ['Total day charge', 'Total eve charge',
                           'Total night charge', 'Total intl charge']
            minute_cols = ['Total day minutes', 'Total eve minutes',
                           'Total night minutes', 'Total intl minutes']
            call_cols = ['Total day calls', 'Total eve calls',
                         'Total night calls', 'Total intl calls']

            if all(col in self.df.columns for col in charge_cols):
                self.df['Total_Charges'] = sum(
                    self.df[col] for col in charge_cols)

            if all(col in self.df.columns for col in minute_cols):
                self.df['Total_Minutes'] = sum(
                    self.df[col] for col in minute_cols)

            if all(col in self.df.columns for col in call_cols):
                self.df['Total_Calls'] = sum(self.df[col] for col in call_cols)

            # Prepare features and target
            self.X = self.df.drop('Churn', axis=1)
            self.y = self.df['Churn']

            print(f"Data prepared successfully: {self.df.shape}")
            print(f"Features: {self.X.shape[1]}")
            print(f"Target distribution: {self.y.value_counts().to_dict()}")

            return True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print(f"Please check if the file exists and has the correct format")
            return False

    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """Split data with stratification"""
        if self.X is None or self.y is None:
            print("Error: Data not loaded. Please load data first.")
            return None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

        return {
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'train_churn_rate': self.y_train.mean() * 100,
            'test_churn_rate': self.y_test.mean() * 100,
            'features': self.X_train.shape[1]
        }

    def compare_algorithms(self):
        """Compare multiple ML algorithms"""
        if self.X_train is None:
            print("Error: Data not split. Please run prepare_train_test_split() first.")
            return None

        algorithms = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': GaussianNB()
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        for name, algorithm in algorithms.items():
            print(f"Training {name}...")

            try:
                # Cross-validation scores
                cv_scores = cross_val_score(algorithm, self.X_train, self.y_train,
                                            cv=cv, scoring='accuracy')

                # Fit model for AUC calculation
                algorithm.fit(self.X_train, self.y_train)
                y_pred_proba = algorithm.predict_proba(self.X_test)[:, 1]
                auc_score = roc_auc_score(self.y_test, y_pred_proba)

                # Store results
                self.model_results[name] = {
                    'model': algorithm,
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'auc_score': auc_score,
                    'cv_scores': cv_scores
                }

                results.append({
                    'Algorithm': name,
                    'CV_Accuracy_Mean': cv_scores.mean(),
                    'CV_Accuracy_Std': cv_scores.std(),
                    'AUC_Score': auc_score
                })

            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue

        if not self.model_results:
            print("Error: No models were trained successfully")
            return None

        # Select best model
        self.best_model_name = max(self.model_results.keys(),
                                   key=lambda x: self.model_results[x]['auc_score'])
        self.best_model = self.model_results[self.best_model_name]['model']

        print(f"Best model: {self.best_model_name}")
        return pd.DataFrame(results)

    def hyperparameter_optimization(self):
        """Optimize hyperparameters for best model"""
        if self.best_model is None:
            print("Error: No best model found. Please run compare_algorithms() first.")
            return {'optimized': False}

        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.1, 0.15],
                'subsample': [0.8, 1.0]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'Decision Tree': {
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        }

        optimization_results = {'optimized': False}

        if self.best_model_name in param_grids:
            print(f"Optimizing {self.best_model_name}...")
            param_grid = param_grids[self.best_model_name]

            try:
                grid_search = GridSearchCV(
                    estimator=self.best_model,
                    param_grid=param_grid,
                    cv=StratifiedKFold(
                        n_splits=3, shuffle=True, random_state=42),
                    scoring='roc_auc',
                    n_jobs=-1
                )

                grid_search.fit(self.X_train, self.y_train)
                self.best_model = grid_search.best_estimator_

                optimization_results = {
                    'optimized': True,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'improvement': grid_search.best_score_ - self.model_results[self.best_model_name]['auc_score']
                }

                # Update model results
                self.model_results[self.best_model_name]['optimized_model'] = self.best_model
                self.model_results[self.best_model_name]['best_params'] = grid_search.best_params_
                self.model_results[self.best_model_name]['optimized_score'] = grid_search.best_score_

            except Exception as e:
                print(f"Error during optimization: {str(e)}")

        return optimization_results

    def evaluate_model(self):
        """Comprehensive model evaluation"""
        if self.best_model is None:
            print("Error: No model to evaluate. Please train models first.")
            return None

        try:
            # Generate predictions
            y_pred = self.best_model.predict(self.X_test)
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc = roc_auc_score(self.y_test, y_pred_proba)

            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            self.evaluation_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1,
                'auc_score': auc,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }

            return self.evaluation_results

        except Exception as e:
            print(f"Error during model evaluation: {str(e)}")
            return None

    def analyze_feature_importance(self):
        """Analyze feature importance"""
        if self.best_model is None:
            return None

        try:
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': self.X.columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)

                return self.feature_importance
            else:
                print(
                    f"Feature importance not available for {self.best_model_name}")
                return None
        except Exception as e:
            print(f"Error analyzing feature importance: {str(e)}")
            return None

    def calculate_business_impact(self):
        """Calculate business impact of the model"""
        if not self.evaluation_results:
            print("Error: No evaluation results available")
            return None

        try:
            tp, fp, fn, tn = (self.evaluation_results['true_positives'],
                              self.evaluation_results['false_positives'],
                              self.evaluation_results['false_negatives'],
                              self.evaluation_results['true_negatives'])

            # Business assumptions
            avg_customer_value = 50
            retention_success_rate = 0.30
            retention_cost_per_customer = 25

            # Calculations
            total_test_customers = len(self.y_test)
            actual_churners = sum(self.y_test)
            customers_correctly_identified = tp
            customers_potentially_saved = int(
                customers_correctly_identified * retention_success_rate)

            # Financial impact
            monthly_revenue_lost = actual_churners * avg_customer_value
            annual_revenue_lost = monthly_revenue_lost * 12
            monthly_revenue_saved = customers_potentially_saved * avg_customer_value
            annual_revenue_saved = monthly_revenue_saved * 12

            # Costs
            customers_targeted = tp + fp
            total_retention_cost = customers_targeted * retention_cost_per_customer
            annual_retention_cost = total_retention_cost * 12

            # Net benefit
            annual_net_benefit = annual_revenue_saved - annual_retention_cost
            roi = (annual_net_benefit / annual_retention_cost) * \
                100 if annual_retention_cost > 0 else 0

            self.business_impact = {
                'total_test_customers': total_test_customers,
                'actual_churners': actual_churners,
                'customers_correctly_identified': customers_correctly_identified,
                'customers_potentially_saved': customers_potentially_saved,
                'annual_revenue_lost': annual_revenue_lost,
                'annual_revenue_saved': annual_revenue_saved,
                'annual_retention_cost': annual_retention_cost,
                'annual_net_benefit': annual_net_benefit,
                'roi_percentage': roi,
                'precision_targeting': (tp/(tp+fp))*100 if (tp+fp) > 0 else 0,
                'churn_prevention_rate': (customers_potentially_saved/actual_churners)*100 if actual_churners > 0 else 0
            }

            return self.business_impact

        except Exception as e:
            print(f"Error calculating business impact: {str(e)}")
            return None

    def create_algorithm_comparison_chart(self):
        """Create algorithm comparison chart"""
        try:
            # Prepare data
            algorithms = list(self.model_results.keys())
            cv_accuracy = [self.model_results[alg]['cv_accuracy_mean']
                           for alg in algorithms]
            cv_std = [self.model_results[alg]['cv_accuracy_std']
                      for alg in algorithms]
            auc_scores = [self.model_results[alg]['auc_score']
                          for alg in algorithms]

            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Cross-Validation Accuracy', 'AUC Scores'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )

            # CV Accuracy with error bars
            colors_cv = ['#e74c3c' if alg ==
                         self.best_model_name else '#3498db' for alg in algorithms]
            fig.add_trace(
                go.Bar(
                    x=algorithms,
                    y=cv_accuracy,
                    error_y=dict(type='data', array=cv_std),
                    name='CV Accuracy',
                    marker_color=colors_cv,
                    hovertemplate='Algorithm: %{x}<br>CV Accuracy: %{y:.3f}<br>Std: %{error_y.array:.3f}<extra></extra>'
                ),
                row=1, col=1
            )

            # AUC Scores
            colors_auc = ['#e74c3c' if alg ==
                          self.best_model_name else '#2ecc71' for alg in algorithms]
            fig.add_trace(
                go.Bar(
                    x=algorithms,
                    y=auc_scores,
                    name='AUC Score',
                    marker_color=colors_auc,
                    hovertemplate='Algorithm: %{x}<br>AUC Score: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )

            fig.update_layout(
                title={
                    'text': f"Algorithm Comparison - Best: {self.best_model_name}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=500,
                showlegend=False
            )

            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(title_text="Score")

            return fig

        except Exception as e:
            print(f"Error creating algorithm comparison chart: {str(e)}")
            return None

    def create_roc_curve_chart(self):
        """Create ROC curve chart"""
        try:
            fpr, tpr, _ = roc_curve(
                self.y_test, self.evaluation_results['prediction_probabilities'])
            auc_score = self.evaluation_results['auc_score']

            fig = go.Figure()

            # ROC curve
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {auc_score:.3f})',
                line=dict(color='#e74c3c', width=3),
                hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
            ))

            # Diagonal line
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='#95a5a6', width=2, dash='dash'),
                hovertemplate='Random Classifier<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': f"ROC Curve - {self.best_model_name}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500,
                width=600
            )

            fig.update_xaxes(range=[0, 1])
            fig.update_yaxes(range=[0, 1])

            return fig

        except Exception as e:
            print(f"Error creating ROC curve chart: {str(e)}")
            return None

    def create_confusion_matrix_chart(self):
        """Create confusion matrix heatmap"""
        try:
            cm = self.evaluation_results['confusion_matrix']

            # Create labels
            labels = ['Retained', 'Churned']

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': f"Confusion Matrix - {self.best_model_name}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=500,
                width=600
            )

            return fig

        except Exception as e:
            print(f"Error creating confusion matrix chart: {str(e)}")
            return None

    def create_performance_metrics_chart(self):
        """Create performance metrics bar chart"""
        try:
            metrics = ['Accuracy', 'Precision', 'Recall',
                       'Specificity', 'F1-Score', 'AUC']
            values = [
                self.evaluation_results['accuracy'],
                self.evaluation_results['precision'],
                self.evaluation_results['recall'],
                self.evaluation_results['specificity'],
                self.evaluation_results['f1_score'],
                self.evaluation_results['auc_score']
            ]

            colors = ['#3498db', '#2ecc71', '#e74c3c',
                      '#f39c12', '#9b59b6', '#1abc9c']

            fig = go.Figure(data=[
                go.Bar(
                    x=metrics,
                    y=values,
                    marker_color=colors,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto',
                    hovertemplate='Metric: %{x}<br>Value: %{y:.3f}<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': f"Model Performance Metrics - {self.best_model_name}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Metrics",
                yaxis_title="Score",
                height=500,
                yaxis=dict(range=[0, 1.1])
            )

            return fig

        except Exception as e:
            print(f"Error creating performance metrics chart: {str(e)}")
            return None

    def create_feature_importance_chart(self):
        """Create feature importance chart"""
        if self.feature_importance is None:
            return None

        try:
            top_features = self.feature_importance.head(15)

            fig = go.Figure(data=[
                go.Bar(
                    y=top_features['feature'],
                    x=top_features['importance'],
                    orientation='h',
                    marker_color='#e74c3c',
                    hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
                )
            ])

            fig.update_layout(
                title={
                    'text': f"Top 15 Feature Importance - {self.best_model_name}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=600,
                yaxis=dict(autorange="reversed")
            )

            return fig

        except Exception as e:
            print(f"Error creating feature importance chart: {str(e)}")
            return None

    def create_business_impact_chart(self):
        """Create business impact visualization"""
        if not self.business_impact:
            return None

        try:
            impact = self.business_impact

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Customer Impact', 'Financial Impact ($)',
                    'Model Effectiveness (%)', 'ROI Analysis'
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )

            # Customer Impact
            customer_metrics = ['Total Customers', 'Actual Churners',
                                'Correctly Identified', 'Potentially Saved']
            customer_values = [
                impact['total_test_customers'],
                impact['actual_churners'],
                impact['customers_correctly_identified'],
                impact['customers_potentially_saved']
            ]

            fig.add_trace(
                go.Bar(
                    x=customer_metrics,
                    y=customer_values,
                    name='Customers',
                    marker_color=['#3498db', '#e74c3c', '#f39c12', '#2ecc71'],
                    hovertemplate='Metric: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=1, col=1
            )

            # Financial Impact
            financial_metrics = ['Revenue Lost',
                                 'Revenue Saved', 'Retention Cost', 'Net Benefit']
            financial_values = [
                impact['annual_revenue_lost'],
                impact['annual_revenue_saved'],
                impact['annual_retention_cost'],
                impact['annual_net_benefit']
            ]

            fig.add_trace(
                go.Bar(
                    x=financial_metrics,
                    y=financial_values,
                    name='Financial ($)',
                    marker_color=['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
                    hovertemplate='Metric: %{x}<br>Amount: $%{y:,.0f}<extra></extra>'
                ),
                row=1, col=2
            )

            # Model Effectiveness
            effectiveness_metrics = ['Precision', 'Churn Prevention Rate']
            effectiveness_values = [
                impact['precision_targeting'],
                impact['churn_prevention_rate']
            ]

            fig.add_trace(
                go.Bar(
                    x=effectiveness_metrics,
                    y=effectiveness_values,
                    name='Effectiveness (%)',
                    marker_color=['#1abc9c', '#e67e22'],
                    hovertemplate='Metric: %{x}<br>Percentage: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )

            # ROI Indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=impact['roi_percentage'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "ROI (%)"},
                    delta={'reference': 100},
                    gauge={
                        'axis': {'range': [None, 500]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 100], 'color': "lightgray"},
                            {'range': [100, 200], 'color': "yellow"},
                            {'range': [200, 500], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 100
                        }
                    }
                ),
                row=2, col=2
            )

            fig.update_layout(
                title={
                    'text': f"Business Impact Analysis - {self.best_model_name}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=700,
                showlegend=False
            )

            return fig

        except Exception as e:
            print(f"Error creating business impact chart: {str(e)}")
            return None

    def generate_all_charts(self):
        """Generate all model training charts and save them"""
        charts = {}

        print("Generating model training charts...")

        try:
            chart_funcs = [
                ('algorithm_comparison', self.create_algorithm_comparison_chart),
                ('roc_curve', self.create_roc_curve_chart),
                ('confusion_matrix', self.create_confusion_matrix_chart),
                ('performance_metrics', self.create_performance_metrics_chart),
                ('feature_importance', self.create_feature_importance_chart),
                ('business_impact', self.create_business_impact_chart)
            ]

            for chart_name, chart_func in chart_funcs:
                try:
                    chart = chart_func()
                    if chart is not None:
                        charts[chart_name] = chart
                        # Save chart as HTML
                        chart_file = os.path.join(
                            self.charts_dir, f"{chart_name}.html")
                        chart.write_html(chart_file)
                        print(
                            f"{chart_name} chart created and saved to {chart_file}")
                    else:
                        print(
                            f"{chart_name} chart could not be created (likely due to missing data)")
                except Exception as e:
                    print(f"Error creating {chart_name} chart: {str(e)}")

            print(
                f"\nChart generation completed. {len(charts)} charts created.")
            return charts

        except Exception as e:
            print(f"Error in generate_all_charts: {str(e)}")
            return charts

    def save_model_and_results(self):
        """Save trained model and results"""
        try:
            # Save the trained model
            model_filename = f'{self.charts_dir}/best_model_{self.best_model_name.replace(" ", "_").lower()}.pkl'
            with open(model_filename, 'wb') as f:
                pickle.dump(self.best_model, f)

            # Save detailed results
            results_summary = {
                'best_model_name': self.best_model_name,
                'model_performance': self.evaluation_results,
                'business_impact': self.business_impact,
                'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
                'all_model_results': {name: {k: v for k, v in results.items() if k != 'model'}
                                      for name, results in self.model_results.items()}
            }

            results_file = f'{self.charts_dir}/model_results_complete.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump(results_summary, f)

            print(f"Model saved to: {model_filename}")
            print(f"Results saved to: {results_file}")

            return model_filename

        except Exception as e:
            print(f"Error saving model and results: {str(e)}")
            return None

    def run_complete_training_pipeline(self):
        """Execute complete model training pipeline"""
        if self.df is None:
            print("No data loaded. Please check data file.")
            return None

        print("\nStarting model training pipeline...")

        try:
            # Pipeline execution with error handling
            split_info = self.prepare_train_test_split()
            if split_info is None:
                print("Failed to split data")
                return None
            print(
                f"Data split completed: {split_info['train_size']} train, {split_info['test_size']} test")

            comparison_results = self.compare_algorithms()
            if comparison_results is None:
                print("Failed to compare algorithms")
                return None
            print(f"Algorithm comparison completed")

            optimization_results = self.hyperparameter_optimization()
            print(f"Hyperparameter optimization completed")

            evaluation_results = self.evaluate_model()
            if evaluation_results is None:
                print("Failed to evaluate model")
                return None
            print(f"Model evaluation completed")

            feature_importance = self.analyze_feature_importance()
            print(f"Feature importance analysis completed")

            business_impact = self.calculate_business_impact()
            if business_impact is None:
                print("Failed to calculate business impact")
                return None
            print(f"Business impact calculation completed")

            charts = self.generate_all_charts()
            print(f"Charts generation completed")

            model_file = self.save_model_and_results()
            print(f"Model and results saved")

            return {
                'split_info': split_info,
                'comparison_results': comparison_results,
                'optimization_results': optimization_results,
                'evaluation_results': evaluation_results,
                'feature_importance': feature_importance,
                'business_impact': business_impact,
                'charts': charts,
                'model_file': model_file
            }

        except Exception as e:
            print(f"Error in training pipeline: {str(e)}")
            return None


# Standalone version
def run_standalone_model_training(data_file="churnbigml80.csv"):
    """Run standalone model training"""
    print("TELECOM CHURN - MODEL TRAINING (FIXED VERSION)")
    print("=" * 55)

    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found")
        print("Please ensure the data file exists in the current directory")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return None, None

    trainer = TelecomModelTrainer(data_file)

    if trainer.df is None:
        print("Error: Could not load data file")
        return None, None

    print(f"Data loaded successfully: {trainer.df.shape}")
    print(f"Features: {trainer.X.shape[1]}")
    print(f"Target distribution: {trainer.y.value_counts().to_dict()}")

    # Run complete pipeline
    results = trainer.run_complete_training_pipeline()

    if results:
        print("\nMODEL TRAINING RESULTS:")
        print("-" * 25)
        print(f"Best Algorithm: {trainer.best_model_name}")
        print(f"Accuracy: {results['evaluation_results']['accuracy']:.3f}")
        print(f"Precision: {results['evaluation_results']['precision']:.3f}")
        print(f"Recall: {results['evaluation_results']['recall']:.3f}")
        print(f"AUC Score: {results['evaluation_results']['auc_score']:.3f}")

        print("\nBUSINESS IMPACT:")
        print("-" * 15)
        print(
            f"Annual Revenue Saved: ${results['business_impact']['annual_revenue_saved']:,.0f}")
        print(
            f"Annual Net Benefit: ${results['business_impact']['annual_net_benefit']:,.0f}")
        print(f"ROI: {results['business_impact']['roi_percentage']:.0f}%")
        print(
            f"Customers Potentially Saved: {results['business_impact']['customers_potentially_saved']}")

        if results['feature_importance'] is not None:
            print("\nTOP 5 FEATURE IMPORTANCE:")
            print("-" * 25)
            for i, row in results['feature_importance'].head(5).iterrows():
                print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")

        print(f"\nModel saved: {results['model_file']}")
        print(f"Charts saved to: {trainer.charts_dir}/")

        # List created chart files
        chart_files = [f for f in os.listdir(
            trainer.charts_dir) if f.endswith('.html')]
        if chart_files:
            print(f"Created chart files: {chart_files}")
        else:
            print("Warning: No chart files were created")

    return trainer, results


# Alternative function to create sample data if your data file is missing
def create_sample_data():
    """Create sample telecom churn data for testing"""
    print("Creating sample data...")

    np.random.seed(42)
    n_samples = 1000

    # Create sample data
    data = {
        'State': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], n_samples),
        'Account length': np.random.normal(100, 40, n_samples).astype(int),
        'Area code': np.random.choice([408, 415, 510], n_samples),
        'International plan': np.random.choice(['Yes', 'No'], n_samples, p=[0.1, 0.9]),
        'Voice mail plan': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Number vmail messages': np.random.poisson(8, n_samples),
        'Total day minutes': np.random.normal(180, 50, n_samples),
        'Total day calls': np.random.normal(100, 20, n_samples).astype(int),
        'Total day charge': np.random.normal(30, 8, n_samples),
        'Total eve minutes': np.random.normal(200, 50, n_samples),
        'Total eve calls': np.random.normal(100, 20, n_samples).astype(int),
        'Total eve charge': np.random.normal(17, 4, n_samples),
        'Total night minutes': np.random.normal(200, 50, n_samples),
        'Total night calls': np.random.normal(100, 20, n_samples).astype(int),
        'Total night charge': np.random.normal(9, 2, n_samples),
        'Total intl minutes': np.random.normal(10, 3, n_samples),
        'Total intl calls': np.random.normal(4, 2, n_samples).astype(int),
        'Total intl charge': np.random.normal(3, 1, n_samples),
        'Customer service calls': np.random.poisson(1, n_samples)
    }

    # Create churn based on some logic
    churn_prob = (
        (data['Customer service calls'] > 3) * 0.4 +
        (data['International plan'] == 'Yes') * 0.2 +
        (data['Total day minutes'] > 250) * 0.2 +
        np.random.random(n_samples) * 0.3
    )

    data['Churn'] = (churn_prob > 0.5).astype(bool)

    df = pd.DataFrame(data)
    df.to_csv('churnbigml80.csv', index=False)
    print("Sample data saved as 'churnbigml80.csv'")
    return df


# Main execution
if __name__ == "__main__":
    # Check if data file exists, if not create sample data
    data_file = "churnbigml80.csv"

    if not os.path.exists(data_file):
        print(f"Data file '{data_file}' not found. Creating sample data...")
        create_sample_data()

    trainer, results = run_standalone_model_training(data_file)

    if trainer and results:
        print("\nModel training completed successfully!")
        print(f"Check the {trainer.charts_dir}/ directory for visualizations")

        # Show what files were created
        if os.path.exists(trainer.charts_dir):
            files = os.listdir(trainer.charts_dir)
            print(f"Files created: {files}")
    else:
        print("\nModel training failed. Please check the error messages above.")
