import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import Dict, List, Tuple
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Cloud Platform Advisor",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .comparison-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #007bff;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class CloudPlatformAdvisor:
    def __init__(self):
        self.cloud_services = {
            'ai_model_training': {
                'AWS': {
                    'advantages': [
                        'SageMaker with comprehensive MLOps pipeline',
                        'P4 instances with 8 A100 GPUs for intensive training',
                        'Extensive library of pre-built algorithms',
                        'Deep integration with AWS ecosystem',
                        'Spot instances for cost-effective training'
                    ],
                    'pricing': 'P4d.24xlarge: ~$32.77/hour\nSageMaker training: $0.269-$40.97/hour\nSpot instances: up to 70% savings',
                    'ai_services': ['SageMaker', 'Bedrock', 'Comprehend', 'Rekognition', 'Textract', 'Polly'],
                    'disadvantages': [
                        'Complex pricing structure with many variables',
                        'Steep learning curve for beginners',
                        'Data transfer costs can accumulate'
                    ],
                    'score': 9.2,
                    'best_for': 'Enterprise ML workflows with complex requirements'
                },
                'Google Cloud': {
                    'advantages': [
                        'Vertex AI unified ML platform',
                        'TPU access for TensorFlow workloads',
                        'Strong integration with TensorFlow ecosystem',
                        'AutoML capabilities for non-experts',
                        'Competitive pricing for compute resources'
                    ],
                    'pricing': 'n1-standard-96 + 8 V100s: ~$24.48/hour\nVertex AI training: $0.20-$30.00/hour\nTPU v3: $1.35/hour',
                    'ai_services': ['Vertex AI', 'AutoML', 'AI Platform', 'TensorFlow Enterprise', 'Vision AI'],
                    'disadvantages': [
                        'Smaller overall ecosystem compared to AWS',
                        'Limited regional availability for some services',
                        'Less third-party integrations'
                    ],
                    'score': 8.7,
                    'best_for': 'TensorFlow-based projects and research workloads'
                },
                'Microsoft Azure': {
                    'advantages': [
                        'Azure Machine Learning studio interface',
                        'Strong enterprise and Microsoft ecosystem integration',
                        'Hybrid cloud capabilities',
                        'Good support for Python, R, and .NET',
                        'Azure OpenAI Service access'
                    ],
                    'pricing': 'NC24rs_v3: ~$22.03/hour\nAzure ML compute: $0.24-$28.80/hour\nLow-priority VMs: up to 80% savings',
                    'ai_services': ['Azure ML', 'Cognitive Services', 'Bot Framework', 'Azure OpenAI', 'Form Recognizer'],
                    'disadvantages': [
                        'Less mature ML ecosystem than AWS/GCP',
                        'Documentation can be fragmented',
                        'Fewer specialized ML instance types'
                    ],
                    'score': 7.8,
                    'best_for': 'Microsoft-centric environments and hybrid scenarios'
                }
            },
            'real_time_analytics': {
                'AWS': {
                    'advantages': [
                        'Kinesis for real-time data streaming',
                        'Redshift for petabyte-scale data warehousing',
                        'QuickSight for business intelligence',
                        'Lambda for serverless stream processing',
                        'MSK (Managed Kafka) for event streaming'
                    ],
                    'pricing': 'Kinesis: $0.015/shard-hour + $0.014/million records\nRedshift: $0.25/hour/node\nLambda: $0.20/million requests',
                    'ai_services': ['Kinesis Analytics', 'SageMaker', 'OpenSearch', 'QuickSight ML'],
                    'disadvantages': [
                        'Complex architecture setup required',
                        'Multiple services needed for complete solution',
                        'Can become expensive at scale'
                    ],
                    'score': 9.1,
                    'best_for': 'Enterprise-scale real-time processing with complex workflows'
                },
                'Google Cloud': {
                    'advantages': [
                        'BigQuery for real-time SQL queries at scale',
                        'Dataflow for unified stream/batch processing',
                        'Pub/Sub for reliable messaging',
                        'Strong integration with analytics tools',
                        'Competitive pricing for data processing'
                    ],
                    'pricing': 'BigQuery: $5/TB processed, $10/TB stored\nDataflow: $0.056/vCPU-hour\nPub/Sub: $0.40/million messages',
                    'ai_services': ['BigQuery ML', 'Dataflow', 'Analytics Hub', 'Looker'],
                    'disadvantages': [
                        'Can get expensive with high query volumes',
                        'Learning curve for BigQuery optimization',
                        'Limited customization in some services'
                    ],
                    'score': 8.6,
                    'best_for': 'SQL-heavy analytics and data science workflows'
                },
                'Microsoft Azure': {
                    'advantages': [
                        'Azure Stream Analytics for real-time processing',
                        'Event Hubs for high-throughput ingestion',
                        'Power BI integration for visualization',
                        'Synapse Analytics for unified analytics',
                        'Strong enterprise security features'
                    ],
                    'pricing': 'Stream Analytics: $0.11/streaming unit-hour\nEvent Hubs: $0.028/million events\nSynapse: $1.20/DWU-hour',
                    'ai_services': ['Stream Analytics', 'Synapse Analytics', 'Power BI', 'Azure Data Explorer'],
                    'disadvantages': [
                        'Less flexibility compared to competitors',
                        'Limited real-time ML capabilities',
                        'Power BI licensing can be complex'
                    ],
                    'score': 7.4,
                    'best_for': 'Microsoft ecosystem integration and enterprise reporting'
                }
            },
            'web_hosting': {
                'AWS': {
                    'advantages': [
                        'S3 + CloudFront for global static site delivery',
                        'EC2 for full control over dynamic sites',
                        'Route 53 for reliable DNS management',
                        'Elastic Load Balancer for high availability',
                        'AWS Amplify for full-stack development'
                    ],
                    'pricing': 'S3: $0.023/GB/month\nCloudFront: $0.085/GB\nEC2 t3.micro: $0.0104/hour (Free tier: 750 hours/month)',
                    'ai_services': ['CloudFront', 'AWS Amplify', 'Lambda@Edge', 'API Gateway'],
                    'disadvantages': [
                        'Complex setup for simple websites',
                        'Many configuration options can overwhelm beginners',
                        'Free tier limitations after 12 months'
                    ],
                    'score': 8.3,
                    'best_for': 'Scalable applications with complex requirements'
                },
                'Google Cloud': {
                    'advantages': [
                        'Firebase hosting with easy deployment',
                        'Cloud Storage for static assets',
                        'Cloud CDN for global content delivery',
                        'App Engine for serverless web apps',
                        'Generous free tier with always-free products'
                    ],
                    'pricing': 'Firebase: Free tier (10GB storage, 125K reads/day)\nCloud Storage: $0.020/GB/month\nApp Engine: Free tier + $0.05/instance-hour',
                    'ai_services': ['Firebase', 'App Engine', 'Cloud Functions', 'Cloud Run'],
                    'disadvantages': [
                        'Fewer enterprise features than AWS',
                        'Limited regions compared to competitors',
                        'Firebase vendor lock-in concerns'
                    ],
                    'score': 8.8,
                    'best_for': 'Rapid prototyping and modern web applications'
                },
                'Microsoft Azure': {
                    'advantages': [
                        'Azure Static Web Apps with GitHub integration',
                        'App Service for managed web hosting',
                        'Azure CDN for content delivery',
                        'Strong Windows/.NET integration',
                        'Built-in authentication and authorization'
                    ],
                    'pricing': 'Static Web Apps: Free tier available\nApp Service: $0.018/hour\nCDN: $0.081/GB + $0.0075/10K requests',
                    'ai_services': ['Static Web Apps', 'App Service', 'Azure Functions', 'Front Door'],
                    'disadvantages': [
                        'Generally more expensive than competitors',
                        'Less documentation for modern JS frameworks',
                        'Windows-centric approach may not suit all developers'
                    ],
                    'score': 7.2,
                    'best_for': '.NET applications and Windows-based development'
                }
            },
            'serverless_computing': {
                'AWS': {
                    'advantages': [
                        'Lambda: industry-leading serverless platform',
                        'Extensive trigger options (200+ event sources)',
                        'Strong ecosystem integration with AWS services',
                        'Step Functions for complex workflow orchestration',
                        'Provisioned concurrency for consistent performance'
                    ],
                    'pricing': 'Lambda: $0.20/million requests + $0.0000166667/GB-second\nAPI Gateway: $3.50/million calls\nStep Functions: $0.025/state transition',
                    'ai_services': ['Lambda', 'API Gateway', 'Step Functions', 'EventBridge', 'SQS'],
                    'disadvantages': [
                        'Cold start latency issues',
                        'Vendor lock-in concerns',
                        '15-minute execution time limit'
                    ],
                    'score': 9.3,
                    'best_for': 'Event-driven architectures and microservices'
                },
                'Google Cloud': {
                    'advantages': [
                        'Cloud Functions for event-driven computing',
                        'Cloud Run for containerized serverless apps',
                        'Generous free tier (2 million invocations/month)',
                        'Fast cold start times',
                        'Strong integration with Firebase'
                    ],
                    'pricing': 'Cloud Functions: $0.40/million invocations + $0.0000025/GB-second\nCloud Run: $0.00002400/vCPU-second + $0.00000250/GB-second',
                    'ai_services': ['Cloud Functions', 'Cloud Run', 'Pub/Sub', 'Workflows'],
                    'disadvantages': [
                        'Fewer integrations compared to AWS Lambda',
                        'Less mature ecosystem',
                        'Limited regional availability'
                    ],
                    'score': 8.1,
                    'best_for': 'Container-based serverless and Firebase integration'
                },
                'Microsoft Azure': {
                    'advantages': [
                        'Azure Functions with multiple hosting options',
                        'Logic Apps for workflow automation',
                        'Strong enterprise integration capabilities',
                        'Durable Functions for stateful scenarios',
                        'KEDA support for event-driven scaling'
                    ],
                    'pricing': 'Functions: $0.20/million executions + $0.000016/GB-second\nLogic Apps: $0.000025/action\nPremium plan: $0.2016/GB-hour',
                    'ai_services': ['Azure Functions', 'Logic Apps', 'Event Grid', 'Service Bus'],
                    'disadvantages': [
                        'Performance inconsistencies reported',
                        'Complex pricing model with multiple tiers',
                        'Less popular in serverless community'
                    ],
                    'score': 7.6,
                    'best_for': 'Enterprise workflows and Microsoft ecosystem integration'
                }
            },
            'data_storage': {
                'AWS': {
                    'advantages': [
                        'S3: industry standard object storage',
                        'Multiple storage classes for cost optimization',
                        'Glacier for long-term archival',
                        'Exceptional durability (99.999999999%)',
                        'Comprehensive data lifecycle management'
                    ],
                    'pricing': 'S3 Standard: $0.023/GB/month\nGlacier Instant Retrieval: $0.004/GB/month\nEBS gp3: $0.08/GB/month',
                    'ai_services': ['S3', 'EBS', 'EFS', 'FSx', 'Glacier', 'Storage Gateway'],
                    'disadvantages': [
                        'Complex pricing with numerous storage classes',
                        'Data transfer costs can accumulate',
                        'Request charges can add up for high-frequency access'
                    ],
                    'score': 9.4,
                    'best_for': 'Large-scale storage with diverse access patterns'
                },
                'Google Cloud': {
                    'advantages': [
                        'Cloud Storage with simple pricing model',
                        'Competitive pricing across all tiers',
                        'Strong integration with BigQuery and AI services',
                        'Multi-regional and dual-regional options',
                        'Automatic redundancy and versioning'
                    ],
                    'pricing': 'Standard: $0.020/GB/month\nNearline: $0.010/GB/month\nColdline: $0.004/GB/month\nArchive: $0.0012/GB/month',
                    'ai_services': ['Cloud Storage', 'Persistent Disk', 'Filestore', 'Firebase Storage'],
                    'disadvantages': [
                        'Fewer storage options than AWS',
                        'Less global presence for edge locations',
                        'Limited hybrid storage solutions'
                    ],
                    'score': 8.2,
                    'best_for': 'Analytics workloads and straightforward storage needs'
                },
                'Microsoft Azure': {
                    'advantages': [
                        'Blob Storage with hot, cool, and archive tiers',
                        'Strong enterprise features and compliance',
                        'Azure Data Lake for analytics',
                        'Excellent hybrid cloud integration',
                        'Zone-redundant storage options'
                    ],
                    'pricing': 'Blob Hot: $0.0184/GB/month\nCool: $0.01/GB/month\nArchive: $0.00099/GB/month\nManaged Disks: $0.048/GB/month',
                    'ai_services': ['Blob Storage', 'Data Lake Storage', 'Managed Disks', 'File Storage'],
                    'disadvantages': [
                        'Generally more expensive than competitors',
                        'Complex tier management and access patterns',
                        'Limited global edge presence'
                    ],
                    'score': 7.7,
                    'best_for': 'Enterprise environments with hybrid requirements'
                }
            }
        }
        
        self.keywords = {
            'ai_model_training': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'model', 'training', 
                                 'neural network', 'deep learning', 'tensorflow', 'pytorch', 'sklearn', 'algorithm',
                                 'computer vision', 'nlp', 'natural language processing', 'data science'],
            'real_time_analytics': ['real-time', 'real time', 'analytics', 'streaming', 'dashboard', 'metrics', 
                                   'monitoring', 'visualization', 'business intelligence', 'bi', 'reporting',
                                   'kafka', 'kinesis', 'pubsub', 'event processing', 'stream processing'],
            'web_hosting': ['website', 'web app', 'web application', 'hosting', 'frontend', 'blog', 'portfolio', 
                           'static site', 'cms', 'wordpress', 'ecommerce', 'landing page', 'spa', 'react', 'vue', 'angular'],
            'serverless_computing': ['serverless', 'lambda', 'function', 'functions', 'microservice', 'microservices',
                                    'event-driven', 'faas', 'function as a service', 'cloud functions', 'azure functions'],
            'data_storage': ['storage', 'database', 'data', 'backup', 'archive', 'file storage', 'object storage',
                            'blob', 's3', 'data lake', 'data warehouse', 'nosql', 'sql', 'mysql', 'postgresql']
        }

    def analyze_requirement(self, user_input: str) -> Tuple[str, Dict]:
        """Analyze user input and return the best matching category and its data."""
        input_lower = user_input.lower()
        scores = {}
        
        for category, keyword_list in self.keywords.items():
            score = 0
            for keyword in keyword_list:
                if keyword in input_lower:
                    # Weight longer keywords more heavily
                    score += len(keyword.split())
            scores[category] = score
        
        # Find the category with the highest score
        best_category = max(scores.keys(), key=lambda k: scores[k]) if any(scores.values()) else 'web_hosting'
        
        return best_category, self.cloud_services[best_category]

    def get_recommendation(self, data: Dict, weights: Dict = None) -> Dict:
        """Get the best recommendation based on scores and user priorities."""
        if weights is None:
            weights = {'cost': 1, 'performance': 1, 'ai_ml': 1}
        
        # Calculate weighted scores
        weighted_data = {}
        for provider, info in data.items():
            base_score = info['score']
            
            # Apply weights based on provider strengths
            cost_multiplier = self.get_cost_multiplier(provider)
            performance_multiplier = self.get_performance_multiplier(provider)
            ai_ml_multiplier = self.get_ai_ml_multiplier(provider)
            
            weighted_score = (
                base_score * 0.4 +  # Base functionality score (40%)
                cost_multiplier * weights['cost'] * 2 +  # Cost factor (20% max)
                performance_multiplier * weights['performance'] * 2 +  # Performance factor (20% max)
                ai_ml_multiplier * weights['ai_ml'] * 2  # AI/ML factor (20% max)
            )
            
            weighted_data[provider] = {
                **info,
                'weighted_score': weighted_score,
                'cost_score': cost_multiplier,
                'performance_score': performance_multiplier,
                'ai_ml_score': ai_ml_multiplier
            }
        
        sorted_providers = sorted(weighted_data.items(), key=lambda x: x[1]['weighted_score'], reverse=True)
        winner = sorted_providers[0]
        
        return {
            'provider': winner[0],
            'score': winner[1]['score'],
            'weighted_score': winner[1]['weighted_score'],
            'reason': winner[1]['best_for'],
            'details': winner[1]
        }
    
    def get_cost_multiplier(self, provider: str) -> float:
        """Get cost effectiveness score for each provider (higher = more cost effective)."""
        cost_scores = {
            'AWS': 3.5,  # Moderate pricing, but complex
            'Google Cloud': 4.5,  # Generally more cost-effective
            'Microsoft Azure': 3.0  # Generally more expensive
        }
        return cost_scores.get(provider, 3.0)
    
    def get_performance_multiplier(self, provider: str) -> float:
        """Get performance score for each provider."""
        performance_scores = {
            'AWS': 4.5,  # Industry leader in performance
            'Google Cloud': 4.2,  # Strong performance, especially for data
            'Microsoft Azure': 3.8  # Good performance, enterprise-focused
        }
        return performance_scores.get(provider, 4.0)
    
    def get_ai_ml_multiplier(self, provider: str) -> float:
        """Get AI/ML capabilities score for each provider."""
        ai_ml_scores = {
            'AWS': 4.5,  # Comprehensive AI/ML services
            'Google Cloud': 4.8,  # Leading in AI/ML innovation
            'Microsoft Azure': 4.0  # Good AI/ML services, growing
        }
        return ai_ml_scores.get(provider, 4.0)

    def create_comparison_dataframe(self, data: Dict) -> pd.DataFrame:
        """Create a DataFrame for easy comparison display."""
        comparison_data = []
        
        for provider, info in data.items():
            comparison_data.append({
                'Provider': provider,
                'Score': info['score'],
                'Key Advantages': ' • '.join(info['advantages'][:3]),
                'Pricing': info['pricing'].split('\n')[0],  # First line only
                'Best For': info['best_for'],
                'Main Services': ', '.join(info['ai_services'][:4])
            })
        
        return pd.DataFrame(comparison_data).sort_values('Score', ascending=False)

    def create_score_chart(self, data: Dict, weights: Dict = None) -> go.Figure:
        """Create a bar chart comparing scores."""
        providers = list(data.keys())
        base_scores = [data[provider]['score'] for provider in providers]
        
        if weights:
            # Calculate weighted scores for comparison
            weighted_scores = []
            for provider in providers:
                cost_mult = self.get_cost_multiplier(provider)
                perf_mult = self.get_performance_multiplier(provider)
                ai_mult = self.get_ai_ml_multiplier(provider)
                
                weighted_score = (
                    data[provider]['score'] * 0.4 +
                    cost_mult * weights['cost'] * 2 +
                    perf_mult * weights['performance'] * 2 +
                    ai_mult * weights['ai_ml'] * 2
                )
                weighted_scores.append(round(weighted_score, 1))
            
            colors = ['#28a745' if provider == 'AWS' else '#17a2b8' if provider == 'Google Cloud' else '#6f42c1' 
                      for provider in providers]
            
            fig = go.Figure()
            
            # Base scores
            fig.add_trace(go.Bar(
                x=providers, 
                y=base_scores, 
                name='Base Score',
                marker_color='lightgray',
                opacity=0.6,
                text=[f"{score}" for score in base_scores],
                textposition='auto'
            ))
            
            # Weighted scores
            fig.add_trace(go.Bar(
                x=providers, 
                y=weighted_scores, 
                name='Weighted Score',
                marker_color=colors,
                text=[f"{score}" for score in weighted_scores],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Platform Comparison: Base vs Weighted Scores",
                xaxis_title="Cloud Provider",
                yaxis_title="Score",
                yaxis=dict(range=[0, max(max(base_scores), max(weighted_scores)) + 1]),
                height=400,
                template="plotly_white",
                barmode='group'
            )
        else:
            colors = ['#28a745' if provider == 'AWS' else '#17a2b8' if provider == 'Google Cloud' else '#6f42c1' 
                      for provider in providers]
            
            fig = go.Figure(data=[
                go.Bar(x=providers, y=base_scores, marker_color=colors, text=base_scores, textposition='auto')
            ])
            
            fig.update_layout(
                title="Platform Comparison Scores",
                xaxis_title="Cloud Provider",
                yaxis_title="Score (out of 10)",
                yaxis=dict(range=[0, 10]),
                height=400,
                template="plotly_white"
            )
        
        return fig

def main():
    # Initialize the advisor
    if 'advisor' not in st.session_state:
        st.session_state.advisor = CloudPlatformAdvisor()
    
    advisor = st.session_state.advisor

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>☁️ Cloud Platform Advisor</h1>
        <p>Get personalized cloud platform recommendations with detailed comparisons</p>
    </div>
    """, unsafe_allow_html=True)

    # Input section
    st.subheader("📝 Describe Your Requirement")
    
    # Main input area
    user_input = st.text_area(
        "What do you need to build or deploy?",
        placeholder="e.g., I need to train a machine learning model for image recognition, or I want to build a real-time dashboard for monitoring user activity...",
        height=150,
        help="Describe your project in natural language. Be specific about your requirements, expected scale, and any special needs."
    )
    
    # Create two columns - left for examples, right for priorities
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**💡 Try these examples:**")
        example_buttons = [
            "Train deep learning models",
            "Real-time data analytics",
            "Host a static website", 
            "Serverless microservices",
            "Store large datasets"
        ]
        
        for example in example_buttons:
            if st.button(f"📋 {example}", key=f"example_{example}", use_container_width=True):
                st.session_state.example_input = example
                st.experimental_rerun()
    
    with col2:
        st.markdown("**⚖️ Set Your Priorities:**")
        st.markdown("*Adjust what factors are most important for your project*")
        
        cost_sensitivity = st.slider(
            "💰 Cost sensitivity (0 = not important, 5 = very important)",
            min_value=0,
            max_value=5,
            value=st.session_state.get('cost_slider', 3),
            key="cost_slider",
            help="How important is cost optimization for your project?"
        )
        
        performance_need = st.slider(
            "⚡ Performance need",
            min_value=0,
            max_value=5,
            value=st.session_state.get('performance_slider', 3),
            key="performance_slider",
            help="How critical is high performance and low latency?"
        )
        
        ai_ml_features = st.slider(
            "🤖 AI/ML features",
            min_value=0,
            max_value=5,
            value=st.session_state.get('ai_ml_slider', 3),
            key="ai_ml_slider",
            help="How important are advanced AI/ML capabilities?"
        )

    # Display current priorities
    if any([cost_sensitivity != 3, performance_need != 3, ai_ml_features != 3]):
        priority_summary = []
        if cost_sensitivity >= 4:
            priority_summary.append("Cost-conscious")
        if performance_need >= 4:
            priority_summary.append("Performance-critical")
        if ai_ml_features >= 4:
            priority_summary.append("AI/ML-focused")
        
        if priority_summary:
            st.info(f"🎯 **Your priorities:** {', '.join(priority_summary)}")

    # Action buttons
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        analyze_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
    with button_col2:
        compare_button = st.button("📊 Compare Clouds", type="secondary", use_container_width=True)

    # Handle example button clicks
    if 'example_input' in st.session_state:
        user_input = st.session_state.example_input
        del st.session_state.example_input
        analyze_button = True

    # Analysis section
    if (analyze_button or compare_button) and user_input.strip():
        # Get user priorities
        weights = {
            'cost': cost_sensitivity,
            'performance': performance_need,
            'ai_ml': ai_ml_features
        }
        
        with st.spinner('🔄 Analyzing your requirements...'):
            time.sleep(1)  # Simulate processing time
            category, data = advisor.analyze_requirement(user_input)
            recommendation = advisor.get_recommendation(data, weights)
            
            # Store results in session state
            st.session_state.analysis_results = {
                'category': category,
                'data': data,
                'recommendation': recommendation,
                'user_input': user_input,
                'weights': weights
            }

    # Display results
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        category = results['category']
        data = results['data']
        recommendation = results['recommendation']
        user_input = results['user_input']
        weights = results.get('weights', {'cost': 3, 'performance': 3, 'ai_ml': 3})
        
        # Results header
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Show recommended cloud in a prominent box
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>✅ Recommended Cloud: {recommendation['provider']}</h3>
            <p><strong>Base Score:</strong> {recommendation['score']}/10 | <strong>Weighted Score:</strong> {recommendation['weighted_score']:.1f}</p>
            <p><strong>Best for:</strong> {recommendation['reason']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"**Your requirement:** {user_input}")
            st.info(f"**Detected category:** {category.replace('_', ' ').title()}")
            
            # Show applied priorities
            active_priorities = []
            if weights['cost'] >= 4:
                active_priorities.append("Cost-sensitive")
            if weights['performance'] >= 4:
                active_priorities.append("Performance-critical")
            if weights['ai_ml'] >= 4:
                active_priorities.append("AI/ML-focused")
            
            if active_priorities:
                st.success(f"**Applied priorities:** {', '.join(active_priorities)}")
        
        with col2:
            # Score chart with weights
            fig = advisor.create_score_chart(data, weights if any(v != 3 for v in weights.values()) else None)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed comparison
        st.subheader("📋 Detailed Comparison")
        
        # Create tabs for each provider
        tabs = st.tabs([f"{provider} ({data[provider]['score']}/10)" for provider in data.keys()])
        
        for i, (provider, info) in enumerate(data.items()):
            with tabs[i]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**✅ Advantages:**")
                    for advantage in info['advantages']:
                        st.markdown(f"• {advantage}")
                    
                    st.markdown("**💰 Pricing:**")
                    st.code(info['pricing'], language=None)
                
                with col2:
                    st.markdown("**🛠️ Key Services:**")
                    service_cols = st.columns(2)
                    for j, service in enumerate(info['ai_services']):
                        with service_cols[j % 2]:
                            st.markdown(f"🔹 {service}")
                    
                    st.markdown("**⚠️ Limitations:**")
                    for disadvantage in info['disadvantages']:
                        st.markdown(f"• {disadvantage}")

        # Comparison table
        st.subheader("📊 Quick Comparison Table")
        df = advisor.create_comparison_dataframe(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Next steps
        st.subheader("🚀 Next Steps")
        next_steps_cols = st.columns(2)
        
        with next_steps_cols[0]:
            st.markdown("""
            **Immediate Actions:**
            1. 🆓 Start with the recommended platform's free tier
            2. 🧮 Use official pricing calculators for accurate estimates
            3. 📚 Review platform-specific documentation
            4. 🏗️ Create a proof-of-concept project
            """)
        
        with next_steps_cols[1]:
            st.markdown("""
            **Strategic Considerations:**
            1. 🔍 Evaluate your team's existing expertise
            2. 🌐 Consider multi-cloud strategies for critical apps
            3. 📈 Plan for future scaling requirements
            4. 🔒 Review security and compliance needs
            """)

        # Platform links
        st.subheader("🔗 Useful Links")
        link_cols = st.columns(3)
        
        with link_cols[0]:
            st.markdown("""
            **AWS Resources:**
            - [AWS Free Tier](https://aws.amazon.com/free/)
            - [AWS Pricing Calculator](https://calculator.aws/)
            - [AWS Documentation](https://docs.aws.amazon.com/)
            """)
        
        with link_cols[1]:
            st.markdown("""
            **Google Cloud Resources:**
            - [GCP Free Tier](https://cloud.google.com/free)
            - [GCP Pricing Calculator](https://cloud.google.com/products/calculator)
            - [GCP Documentation](https://cloud.google.com/docs)
            """)
        
        with link_cols[2]:
            st.markdown("""
            **Azure Resources:**
            - [Azure Free Account](https://azure.microsoft.com/free/)
            - [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/)
            - [Azure Documentation](https://docs.microsoft.com/azure/)
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>💡 <strong>Pro Tip:</strong> Consider starting with free tiers to test your use case before committing to paid plans.</p>
        <p>🔄 This tool uses keyword matching for analysis. For complex requirements, consider consulting with cloud architects.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
