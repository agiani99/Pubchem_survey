import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="PubChem Temporal Trends Analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F18F01;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and merge the datasets"""
    try:
        # Load both datasets
        creation_dates = pd.read_csv('pubchem_creation_dates.csv')
        molecular_analysis = pd.read_csv('molecular_analysis.csv')
        
        # Merge on CID
        merged_data = pd.merge(creation_dates, molecular_analysis, on='CID', how='inner')
        
        # Clean up temporal periods and create ordered categories
        merged_data['Temporal_Period'] = merged_data['Temporal_Period'].str.strip()
        
        # Define temporal order
        period_order = [
            "1996-2000", "2000-2004", "2004-2008", "2008-2010", 
            "2010-2012", "2012-2015", "2015-2017", "2017-2020", "2020-2024"
        ]
        
        # Filter out unknown periods and order
        merged_data = merged_data[merged_data['Temporal_Period'].isin(period_order)]
        merged_data['Temporal_Period'] = pd.Categorical(
            merged_data['Temporal_Period'], 
            categories=period_order, 
            ordered=True
        )
        
        # Extract period start year for numeric analysis
        merged_data['Period_Start_Year'] = merged_data['Temporal_Period'].astype(str).str.extract('(\d{4})').astype(int)
        
        return merged_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_temporal_statistics(df, descriptor):
    """Calculate temporal statistics for a descriptor"""
    stats_data = []
    
    for period in df['Temporal_Period'].cat.categories:
        period_data = df[df['Temporal_Period'] == period][descriptor].dropna()
        
        if len(period_data) > 0:
            stats_data.append({
                'Temporal_Period': period,
                'Count': len(period_data),
                'Mean': period_data.mean(),
                'Median': period_data.median(),
                'Std': period_data.std(),
                'Min': period_data.min(),
                'Max': period_data.max(),
                'Q25': period_data.quantile(0.25),
                'Q75': period_data.quantile(0.75)
            })
    
    return pd.DataFrame(stats_data)

def create_trend_plot(df, descriptor, plot_type='line'):
    """Create trend plots for descriptors over time"""
    stats_df = calculate_temporal_statistics(df, descriptor)
    
    if plot_type == 'line':
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=stats_df['Temporal_Period'],
            y=stats_df['Mean'],
            mode='lines+markers',
            name='Mean',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=stats_df['Temporal_Period'],
            y=stats_df['Median'],
            mode='lines+markers',
            name='Median',
            line=dict(color='#A23B72', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=stats_df['Temporal_Period'],
            y=stats_df['Q75'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=stats_df['Temporal_Period'],
            y=stats_df['Q25'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(46, 134, 171, 0.2)',
            name='IQR',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f'{descriptor} Trends Over Time',
            xaxis_title='Temporal Period',
            yaxis_title=descriptor,
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
    elif plot_type == 'box':
        fig = px.box(
            df, 
            x='Temporal_Period', 
            y=descriptor,
            title=f'{descriptor} Distribution by Temporal Period',
            template='plotly_white'
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Temporal Period',
            yaxis_title=descriptor
        )
        
        fig.update_xaxes(tickangle=45)
    
    return fig

def calculate_correlation_with_time(df):
    """Calculate correlations between descriptors and time"""
    descriptors = ['MW', 'RotBonds', 'TPSA', 'HAcceptors', 'HDonors', 'cLogP',
                  'AromaticRings', 'TotalRings', 'Heterocycles', 'NumF', 'NumCl',
                  'ChiralCenters', 'QED']
    
    correlations = []
    
    for desc in descriptors:
        if desc in df.columns:
            # Calculate Pearson correlation with period start year
            corr_coef, p_value = stats.pearsonr(
                df['Period_Start_Year'].dropna(), 
                df[desc].dropna()
            )
            
            # Calculate Spearman correlation (rank-based)
            spearman_coef, spearman_p = stats.spearmanr(
                df['Period_Start_Year'].dropna(), 
                df[desc].dropna()
            )
            
            correlations.append({
                'Descriptor': desc,
                'Pearson_r': corr_coef,
                'Pearson_p': p_value,
                'Spearman_r': spearman_coef,
                'Spearman_p': spearman_p,
                'Significant': p_value < 0.05
            })
    
    return pd.DataFrame(correlations)

def perform_pca_analysis(df):
    """Perform PCA analysis on molecular descriptors by time period"""
    descriptors = ['MW', 'RotBonds', 'TPSA', 'HAcceptors', 'HDonors', 'cLogP',
                  'AromaticRings', 'TotalRings', 'Heterocycles', 'NumF', 'NumCl',
                  'ChiralCenters', 'QED']
    
    # Filter available descriptors
    available_descriptors = [d for d in descriptors if d in df.columns]
    
    # Prepare data
    pca_data = df[available_descriptors + ['Temporal_Period']].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_data[available_descriptors])
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA results dataframe
    pca_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    pca_df['Temporal_Period'] = pca_data['Temporal_Period'].values
    
    return pca_df, pca, available_descriptors

def perform_tsne_analysis(df, sample_size=1000):
    """Perform t-SNE analysis on molecular descriptors by time period"""
    descriptors = ['MW', 'RotBonds', 'TPSA', 'HAcceptors', 'HDonors', 'cLogP',
                  'AromaticRings', 'TotalRings', 'Heterocycles', 'NumF', 'NumCl',
                  'ChiralCenters', 'QED']
    
    # Filter available descriptors
    available_descriptors = [d for d in descriptors if d in df.columns]
    
    # Prepare data - sample for t-SNE performance
    tsne_data = df[available_descriptors + ['Temporal_Period']].dropna()
    
    # Sample data if too large (t-SNE is computationally expensive)
    if len(tsne_data) > sample_size:
        tsne_data = tsne_data.sample(n=sample_size, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(tsne_data[available_descriptors])
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create t-SNE results dataframe
    tsne_df = pd.DataFrame(X_tsne, columns=['tSNE1', 'tSNE2'])
    tsne_df['Temporal_Period'] = tsne_data['Temporal_Period'].values
    tsne_df['CID'] = tsne_data.index
    
    return tsne_df, available_descriptors
    """Perform PCA analysis on molecular descriptors by time period"""
    descriptors = ['MW', 'RotBonds', 'TPSA', 'HAcceptors', 'HDonors', 'cLogP',
                  'AromaticRings', 'TotalRings', 'Heterocycles', 'NumF', 'NumCl',
                  'ChiralCenters', 'QED']
    
    # Filter available descriptors
    available_descriptors = [d for d in descriptors if d in df.columns]
    
    # Prepare data
    pca_data = df[available_descriptors + ['Temporal_Period']].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pca_data[available_descriptors])
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA results dataframe
    pca_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    pca_df['Temporal_Period'] = pca_data['Temporal_Period'].values
    
    return pca_df, pca, available_descriptors

def main():
    st.markdown('<h1 class="main-header">🧪 PubChem Temporal Trends Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This application analyzes temporal trends in molecular descriptors from PubChem compounds,
    showing how chemical space has evolved over different time periods.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure both CSV files are in the correct location.")
        return
    
    st.success(f"✅ Data loaded successfully! Total compounds: {len(df):,}")
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Analysis Controls")
    
    # Available descriptors
    descriptors = ['MW', 'RotBonds', 'TPSA', 'HAcceptors', 'HDonors', 'cLogP',
                  'AromaticRings', 'TotalRings', 'Heterocycles', 'NumF', 'NumCl',
                  'ChiralCenters', 'QED']
    
    available_descriptors = [d for d in descriptors if d in df.columns]
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Individual Descriptor Trends", "Correlation Analysis", 
         "PCA Analysis", "t-SNE Analysis", "Statistical Summary", "Period Comparison"]
    )
    
    # Main content area
    if analysis_type == "Overview":
        st.markdown('<h2 class="section-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Compounds", f"{len(df):,}")
        
        with col2:
            st.metric("Time Periods", len(df['Temporal_Period'].unique()))
        
        with col3:
            st.metric("Descriptors", len(available_descriptors))
        
        with col4:
            date_range = f"{df['Temporal_Period'].min()} - {df['Temporal_Period'].max()}"
            st.metric("Date Range", date_range)
        
        # Sample distribution by period
        st.subheader("Sample Distribution by Temporal Period")
        period_counts = df['Temporal_Period'].value_counts().sort_index()
        
        fig = px.bar(
            x=period_counts.index.astype(str),
            y=period_counts.values,
            title="Number of Compounds by Temporal Period",
            template='plotly_white',
            color=period_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title="Temporal Period",
            yaxis_title="Number of Compounds",
            height=400,
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick statistics
        st.subheader("Quick Statistics by Period")
        quick_stats = df.groupby('Temporal_Period').agg({
            'MW': ['mean', 'std'],
            'cLogP': ['mean', 'std'],
            'QED': ['mean', 'std']
        }).round(2)
        
        quick_stats.columns = ['_'.join(col).strip() for col in quick_stats.columns]
        st.dataframe(quick_stats, use_container_width=True)
    
    elif analysis_type == "Individual Descriptor Trends":
        st.markdown('<h2 class="section-header">📈 Individual Descriptor Trends</h2>', unsafe_allow_html=True)
        
        # Descriptor selection
        selected_descriptor = st.sidebar.selectbox(
            "Select Descriptor",
            available_descriptors,
            index=0 if 'MW' in available_descriptors else 0
        )
        
        # Plot type selection
        plot_type = st.sidebar.radio(
            "Plot Type",
            ["line", "box"]
        )
        
        # Show statistics
        if st.sidebar.checkbox("Show Statistics"):
            stats_df = calculate_temporal_statistics(df, selected_descriptor)
            st.subheader(f"Statistics for {selected_descriptor}")
            st.dataframe(stats_df, use_container_width=True)
        
        # Create and display plot
        fig = create_trend_plot(df, selected_descriptor, plot_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        st.subheader("Trend Analysis")
        
        # Calculate trend
        stats_df = calculate_temporal_statistics(df, selected_descriptor)
        if len(stats_df) > 1:
            x_vals = np.arange(len(stats_df))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, stats_df['Mean'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                trend_direction = "Increasing" if slope > 0 else "Decreasing"
                st.metric("Trend Direction", trend_direction)
            with col2:
                st.metric("R²", f"{r_value**2:.3f}")
            with col3:
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("Statistical Significance", significance)
    
    elif analysis_type == "Correlation Analysis":
        st.markdown('<h2 class="section-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
        
        # Calculate correlations
        corr_df = calculate_correlation_with_time(df)
        
        # Display correlation table
        st.subheader("Correlations with Time")
        
        # Sort by absolute correlation
        corr_df_display = corr_df.copy()
        corr_df_display['Abs_Pearson_r'] = abs(corr_df_display['Pearson_r'])
        corr_df_display = corr_df_display.sort_values('Abs_Pearson_r', ascending=False)
        
        # Color coding for significance
        def highlight_significant(val):
            if val:
                return 'background-color: #90EE90'
            else:
                return 'background-color: #FFB6C1'
        
        styled_df = corr_df_display.style.applymap(
            highlight_significant, subset=['Significant']
        ).format({
            'Pearson_r': '{:.3f}',
            'Pearson_p': '{:.3e}',
            'Spearman_r': '{:.3f}',
            'Spearman_p': '{:.3e}'
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_df['Pearson_r'].values.reshape(1, -1),
            x=corr_df['Descriptor'].values,
            y=['Pearson Correlation'],
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation Coefficient")
        ))
        
        fig.update_layout(
            title="Temporal Correlations with Molecular Descriptors",
            height=300,
            template='plotly_white'
        )
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top correlations
        st.subheader("Strongest Temporal Trends")
        top_positive = corr_df.nlargest(3, 'Pearson_r')[['Descriptor', 'Pearson_r']]
        top_negative = corr_df.nsmallest(3, 'Pearson_r')[['Descriptor', 'Pearson_r']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Increasing Over Time:**")
            for _, row in top_positive.iterrows():
                st.write(f"- {row['Descriptor']}: r = {row['Pearson_r']:.3f}")
        
        with col2:
            st.write("**Most Decreasing Over Time:**")
            for _, row in top_negative.iterrows():
                st.write(f"- {row['Descriptor']}: r = {row['Pearson_r']:.3f}")
    
    elif analysis_type == "PCA Analysis":
        st.markdown('<h2 class="section-header">🎯 PCA Analysis</h2>', unsafe_allow_html=True)
        
        # Perform PCA
        with st.spinner("Performing PCA analysis..."):
            pca_df, pca, descriptors_used = perform_pca_analysis(df)
        
        # Explained variance
        st.subheader("Explained Variance")
        
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(min(10, len(explained_var)))],
                y=explained_var[:10],
                name='Individual'
            ))
            fig.update_layout(
                title="Explained Variance by Principal Component",
                xaxis_title="Principal Component",
                yaxis_title="Explained Variance Ratio",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(min(10, len(cumulative_var)))],
                y=cumulative_var[:10],
                mode='lines+markers',
                name='Cumulative'
            ))
            fig.update_layout(
                title="Cumulative Explained Variance",
                xaxis_title="Principal Component",
                yaxis_title="Cumulative Explained Variance",
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # PCA scatter plot
        st.subheader("PCA Scatter Plot by Temporal Period")
        
        fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Temporal_Period',
            title='PCA: Chemical Space Evolution Over Time',
            template='plotly_white',
            hover_data=['PC3'],
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Component loadings
        st.subheader("Principal Component Loadings")
        
        n_components_show = st.slider("Number of components to show", 1, 5, 3)
        
        loadings_df = pd.DataFrame(
            pca.components_[:n_components_show].T,
            columns=[f'PC{i+1}' for i in range(n_components_show)],
            index=descriptors_used
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=loadings_df.T.values,
            x=loadings_df.index,
            y=loadings_df.columns,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Loading Value"),
            hovertemplate='Descriptor: %{x}<br>Component: %{y}<br>Loading: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="PCA Component Loadings",
            xaxis_title="Molecular Descriptors",
            yaxis_title="Principal Components",
            template='plotly_white'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "t-SNE Analysis":
        st.markdown('<h2 class="section-header">🌟 t-SNE Analysis</h2>', unsafe_allow_html=True)
        
        st.info("t-SNE (t-Distributed Stochastic Neighbor Embedding) reveals local structure and clusters in chemical space.")
        
        # t-SNE parameters
        st.sidebar.subheader("t-SNE Parameters")
        sample_size = st.sidebar.slider("Sample Size (for performance)", 500, 5000, 1000, 100)
        perplexity = st.sidebar.slider("Perplexity", 5, 50, 30, 5)
        n_iter = st.sidebar.slider("Iterations", 500, 2000, 1000, 250)
        
        # Perform t-SNE
        with st.spinner("Performing t-SNE analysis... This may take a moment."):
            # Custom t-SNE with user parameters
            descriptors = ['MW', 'RotBonds', 'TPSA', 'HAcceptors', 'HDonors', 'cLogP',
                          'AromaticRings', 'TotalRings', 'Heterocycles', 'NumF', 'NumCl',
                          'ChiralCenters', 'QED']
            
            available_descriptors = [d for d in descriptors if d in df.columns]
            
            # Prepare data
            tsne_data = df[available_descriptors + ['Temporal_Period']].dropna()
            
            if len(tsne_data) > sample_size:
                tsne_data = tsne_data.sample(n=sample_size, random_state=42)
                st.info(f"Sampled {sample_size} compounds from {len(df)} total for t-SNE performance")
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(tsne_data[available_descriptors])
            
            # Perform t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
            X_tsne = tsne.fit_transform(X_scaled)
            
            # Create results dataframe
            tsne_df = pd.DataFrame(X_tsne, columns=['tSNE1', 'tSNE2'])
            tsne_df['Temporal_Period'] = tsne_data['Temporal_Period'].values
        
        st.success(f"t-SNE completed! Analyzed {len(tsne_df)} compounds.")
        
        # t-SNE scatter plot colored by temporal period
        st.subheader("t-SNE: Chemical Space Clusters by Temporal Period")
        
        fig = px.scatter(
            tsne_df,
            x='tSNE1',
            y='tSNE2',
            color='Temporal_Period',
            title='t-SNE: Chemical Space Evolution Over Time',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(
            height=700,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Period distribution analysis
        st.subheader("Temporal Period Distribution in t-SNE Space")
        
        period_counts = tsne_df['Temporal_Period'].value_counts().sort_index()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Sample Distribution:**")
            for period, count in period_counts.items():
                percentage = (count / len(tsne_df)) * 100
                st.write(f"• {period}: {count} ({percentage:.1f}%)")
        
        with col2:
            # Density plot by period
            fig = go.Figure()
            
            for period in tsne_df['Temporal_Period'].unique():
                period_data = tsne_df[tsne_df['Temporal_Period'] == period]
                
                fig.add_trace(go.Scatter(
                    x=period_data['tSNE1'],
                    y=period_data['tSNE2'],
                    mode='markers',
                    name=period,
                    marker=dict(
                        size=6,
                        opacity=0.7
                    ),
                    showlegend=False
                ))
            
            fig.update_layout(
                title="t-SNE Overlay View",
                xaxis_title="t-SNE Component 1",
                yaxis_title="t-SNE Component 2",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster analysis
        st.subheader("Chemical Space Interpretation")
        
        # Calculate centroids for each period
        centroids = tsne_df.groupby('Temporal_Period')[['tSNE1', 'tSNE2']].mean()
        
        st.write("""
        **Key Observations:**
        - **Clusters** indicate groups of chemically similar compounds
        - **Separation** between periods suggests evolving chemical preferences
        - **Mixed regions** show consistent chemical classes across time
        - **Isolated clusters** may represent period-specific compound types
        """)
        
        # Period centroids table
        st.subheader("Temporal Period Centroids")
        centroids_display = centroids.round(3)
        centroids_display['Period'] = centroids_display.index
        centroids_display = centroids_display[['Period', 'tSNE1', 'tSNE2']]
        
        st.dataframe(centroids_display, use_container_width=True)
        
        # Download option
        csv_data = tsne_df.to_csv(index=False)
        st.download_button(
            label="Download t-SNE Results as CSV",
            data=csv_data,
            file_name="tsne_temporal_analysis.csv",
            mime="text/csv"
        )
    
    elif analysis_type == "Statistical Summary":
        st.markdown('<h2 class="section-header">📋 Statistical Summary</h2>', unsafe_allow_html=True)
        
        # Descriptor selection for detailed analysis
        selected_descriptors = st.sidebar.multiselect(
            "Select Descriptors for Analysis",
            available_descriptors,
            default=available_descriptors[:5]
        )
        
        if selected_descriptors:
            # Create comprehensive statistics table
            summary_stats = []
            
            for desc in selected_descriptors:
                for period in df['Temporal_Period'].cat.categories:
                    period_data = df[df['Temporal_Period'] == period][desc].dropna()
                    
                    if len(period_data) > 0:
                        summary_stats.append({
                            'Descriptor': desc,
                            'Period': period,
                            'Count': len(period_data),
                            'Mean': period_data.mean(),
                            'Median': period_data.median(),
                            'Std': period_data.std(),
                            'CV': period_data.std() / period_data.mean() if period_data.mean() != 0 else np.nan,
                            'Min': period_data.min(),
                            'Max': period_data.max(),
                            'Range': period_data.max() - period_data.min(),
                            'Skewness': stats.skew(period_data),
                            'Kurtosis': stats.kurtosis(period_data)
                        })
            
            summary_df = pd.DataFrame(summary_stats)
            
            # Display summary
            st.subheader("Comprehensive Statistical Summary")
            
            # Format for display
            formatted_df = summary_df.copy()
            numeric_columns = ['Mean', 'Median', 'Std', 'CV', 'Min', 'Max', 'Range', 'Skewness', 'Kurtosis']
            for col in numeric_columns:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].round(3)
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Download option
            csv_data = formatted_df.to_csv(index=False)
            st.download_button(
                label="Download Statistical Summary as CSV",
                data=csv_data,
                file_name="molecular_descriptors_statistical_summary.csv",
                mime="text/csv"
            )
    
    elif analysis_type == "Period Comparison":
        st.markdown('<h2 class="section-header">⚖️ Period Comparison</h2>', unsafe_allow_html=True)
        
        # Period selection
        periods = list(df['Temporal_Period'].cat.categories)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            period1 = st.selectbox("Select First Period", periods, index=0)
        with col2:
            period2 = st.selectbox("Select Second Period", periods, index=len(periods)-1 if len(periods) > 1 else 0)
        
        # Descriptor selection
        descriptor = st.sidebar.selectbox(
            "Select Descriptor for Comparison",
            available_descriptors
        )
        
        if period1 != period2:
            # Get data for both periods
            data1 = df[df['Temporal_Period'] == period1][descriptor].dropna()
            data2 = df[df['Temporal_Period'] == period2][descriptor].dropna()
            
            # Statistical test
            if len(data1) > 0 and len(data2) > 0:
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                
                # Effect size (Cohen's d)
                cohens_d = (data1.mean() - data2.mean()) / np.sqrt(((len(data1)-1)*data1.var() + (len(data2)-1)*data2.var()) / (len(data1)+len(data2)-2))
                
                # Display comparison results
                st.subheader(f"Comparison: {period1} vs {period2}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(f"{period1} Mean", f"{data1.mean():.3f}")
                with col2:
                    st.metric(f"{period2} Mean", f"{data2.mean():.3f}")
                with col3:
                    st.metric("P-value", f"{p_value:.2e}")
                with col4:
                    effect_size = "Small" if abs(cohens_d) < 0.5 else "Medium" if abs(cohens_d) < 0.8 else "Large"
                    st.metric("Effect Size", effect_size)
                
                # Distribution comparison plot
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=data1,
                    name=period1,
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.add_trace(go.Histogram(
                    x=data2,
                    name=period2,
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig.update_layout(
                    title=f'{descriptor} Distribution Comparison',
                    xaxis_title=descriptor,
                    yaxis_title='Frequency',
                    barmode='overlay',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical interpretation
                st.subheader("Statistical Interpretation")
                
                significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
                direction = "higher" if data2.mean() > data1.mean() else "lower"
                
                st.write(f"""
                **Results Summary:**
                - The difference in {descriptor} between {period1} and {period2} is **{significance}** (p = {p_value:.2e})
                - {period2} shows **{direction}** values compared to {period1}
                - Effect size: **{effect_size}** (Cohen's d = {cohens_d:.3f})
                - Sample sizes: {period1} (n = {len(data1)}), {period2} (n = {len(data2)})
                """)
        else:
            st.warning("Please select two different periods for comparison.")
    
    # Footer
    st.markdown("---")
    st.markdown("**📊 Data Sources:** PubChem Creation Dates & Molecular Analysis")
    st.markdown("**🔬 Generated by:** PubChem Temporal Trends Analysis Tool")

if __name__ == "__main__":
    main()