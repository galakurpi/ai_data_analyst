"""
Advanced Analytics Module for AI Data Analyst

This module provides business intelligence tools that the AI can automatically
select based on the user's question. Each tool returns both data AND actionable insights.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime, timedelta


# =============================================================================
# ANALYSIS TOOL DEFINITIONS (for OpenAI function calling)
# =============================================================================

ANALYSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rfm_analysis",
            "description": "Perform RFM (Recency, Frequency, Monetary) customer segmentation analysis. Use this when the user wants to understand customer segments, identify best customers, at-risk customers, or customer value distribution. Great for questions about 'who are my best customers', 'customer segments', 'customer loyalty'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps answer the user's question"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sales_trend_analysis",
            "description": "Analyze sales trends over time with moving averages, growth rates, and seasonality detection. Use for questions about 'how are sales trending', 'growth rate', 'seasonal patterns', 'sales over time', 'performance trends'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly"],
                        "description": "Time period for aggregation"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps"
                    }
                },
                "required": ["period", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "product_performance_analysis",
            "description": "Analyze product performance with Pareto analysis (80/20 rule), identify top performers, underperformers, and category insights. Use for 'best selling products', 'product performance', 'which products should I focus on', 'inventory decisions'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["revenue", "quantity", "orders"],
                        "description": "Metric to analyze products by"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps"
                    }
                },
                "required": ["metric", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "correlation_analysis",
            "description": "Find correlations and relationships between different metrics (price vs quantity, category vs revenue, etc). Use for 'what affects sales', 'relationships between', 'what drives revenue', 'pricing analysis'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pca_analysis",
            "description": "Principal Component Analysis to find hidden patterns and reduce dimensionality. Reveals which factors explain most variance in the data. Use for 'hidden patterns', 'what factors matter most', 'key drivers', 'dimensionality reduction', 'explain variance'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "enum": ["customers", "products"],
                        "description": "What entity to analyze"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps"
                    }
                },
                "required": ["target", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "customer_clustering",
            "description": "Cluster customers into distinct groups based on behavior patterns using K-means. Use for 'customer personas', 'group customers', 'behavioral segments', 'marketing segments'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_clusters": {
                        "type": "integer",
                        "description": "Number of customer segments to create (3-6 recommended)"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps"
                    }
                },
                "required": ["n_clusters", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cohort_analysis",
            "description": "Analyze customer retention by signup cohort. Shows how different customer cohorts behave over time. Use for 'retention', 'customer lifetime', 'cohort behavior', 'when do customers drop off'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "geographic_analysis",
            "description": "Analyze performance by geographic region/country. Use for 'sales by country', 'regional performance', 'geographic distribution', 'market analysis'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this analysis helps"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simple_query",
            "description": "Execute a simple SQL query and visualization for straightforward questions that don't need advanced analysis. Use for basic questions like 'show me sales', 'list products', 'count orders', simple aggregations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]


# =============================================================================
# ANALYSIS IMPLEMENTATIONS
# =============================================================================

def rfm_analysis(conn):
    """
    RFM (Recency, Frequency, Monetary) Analysis
    Segments customers based on their purchase behavior.
    """
    query = """
    SELECT 
        c.id as customer_id,
        c.name as customer_name,
        c.country,
        MAX(o.date) as last_order_date,
        COUNT(DISTINCT o.id) as frequency,
        SUM(oi.amount) as monetary
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.status != 'Returned'
    GROUP BY c.id, c.name, c.country
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Calculate Recency (days since last order)
    df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    today = pd.Timestamp.now().normalize()
    df['recency'] = (today - df['last_order_date']).dt.days
    
    # Score each dimension (1-5, where 5 is best)
    # For recency, lower is better so we reverse
    df['R_score'] = pd.qcut(df['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    df['F_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    df['M_score'] = pd.qcut(df['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    
    # Combined RFM score
    df['RFM_score'] = df['R_score'] + df['F_score'] + df['M_score']
    
    # Segment customers
    def segment_customer(row):
        if row['RFM_score'] >= 13:
            return 'Champions'
        elif row['RFM_score'] >= 10:
            return 'Loyal Customers'
        elif row['R_score'] >= 4 and row['F_score'] <= 2:
            return 'New Customers'
        elif row['R_score'] <= 2 and row['F_score'] >= 3:
            return 'At Risk'
        elif row['RFM_score'] <= 6:
            return 'Lost'
        else:
            return 'Potential Loyalists'
    
    df['segment'] = df.apply(segment_customer, axis=1)
    
    # Create visualizations
    # 1. Segment distribution pie chart
    segment_counts = df['segment'].value_counts().reset_index()
    segment_counts.columns = ['segment', 'count']
    
    fig1 = px.pie(
        segment_counts, 
        values='count', 
        names='segment',
        title='Customer Segments Distribution',
        color='segment',
        color_discrete_map={
            'Champions': '#10b981',
            'Loyal Customers': '#3b82f6',
            'Potential Loyalists': '#8b5cf6',
            'New Customers': '#06b6d4',
            'At Risk': '#f59e0b',
            'Lost': '#ef4444'
        }
    )
    
    # 2. RFM scatter plot
    fig2 = px.scatter(
        df,
        x='frequency',
        y='monetary',
        color='segment',
        size='RFM_score',
        hover_data=['customer_name', 'recency', 'country'],
        title='Customer Value Map (Frequency vs Monetary)',
        labels={'frequency': 'Number of Orders', 'monetary': 'Total Spend ($)'},
        color_discrete_map={
            'Champions': '#10b981',
            'Loyal Customers': '#3b82f6',
            'Potential Loyalists': '#8b5cf6',
            'New Customers': '#06b6d4',
            'At Risk': '#f59e0b',
            'Lost': '#ef4444'
        }
    )
    
    # Generate insights
    segment_summary = df.groupby('segment').agg({
        'customer_id': 'count',
        'monetary': 'sum',
        'frequency': 'mean',
        'recency': 'mean'
    }).round(2)
    
    champions = df[df['segment'] == 'Champions']
    at_risk = df[df['segment'] == 'At Risk']
    
    insights = {
        "title": "RFM Customer Segmentation Analysis",
        "key_findings": [
            f"🏆 You have {len(champions)} Champion customers who generate ${champions['monetary'].sum():,.0f} in revenue",
            f"⚠️ {len(at_risk)} customers are At Risk - they used to buy frequently but haven't ordered recently",
            f"📊 Top 20% of customers contribute {(df.nlargest(int(len(df)*0.2), 'monetary')['monetary'].sum() / df['monetary'].sum() * 100):.0f}% of total revenue",
        ],
        "recommendations": [
            "🎯 **Champions**: Reward them with VIP perks, early access to new products",
            "🔔 **At Risk**: Send win-back campaigns with special offers immediately",
            "💎 **Potential Loyalists**: Upsell and cross-sell to increase their value",
            "👋 **New Customers**: Focus on onboarding and first-purchase experience",
        ],
        "segment_summary": segment_summary.to_dict()
    }
    
    return {
        "charts": [fig1.to_json(), fig2.to_json()],
        "insights": insights,
        "data_preview": df[['customer_name', 'segment', 'recency', 'frequency', 'monetary', 'RFM_score']].head(10).to_dict('records')
    }


def sales_trend_analysis(conn, period='monthly'):
    """
    Analyze sales trends with moving averages and growth rates.
    """
    query = """
    SELECT 
        o.date,
        SUM(oi.amount) as revenue,
        COUNT(DISTINCT o.id) as orders,
        SUM(oi.quantity) as units_sold
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.status != 'Returned'
    GROUP BY o.date
    ORDER BY o.date
    """
    
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    
    # Resample based on period
    period_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'M'}
    df_resampled = df.set_index('date').resample(period_map[period]).sum().reset_index()
    
    # Calculate moving averages and growth
    df_resampled['MA_7'] = df_resampled['revenue'].rolling(window=3, min_periods=1).mean()
    df_resampled['growth_rate'] = df_resampled['revenue'].pct_change() * 100
    
    # Create trend visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Revenue Trend with Moving Average', 'Growth Rate (%)'),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Revenue line
    fig.add_trace(
        go.Scatter(x=df_resampled['date'], y=df_resampled['revenue'], 
                   name='Revenue', mode='lines+markers',
                   line=dict(color='#3b82f6', width=2)),
        row=1, col=1
    )
    
    # Moving average
    fig.add_trace(
        go.Scatter(x=df_resampled['date'], y=df_resampled['MA_7'],
                   name='Moving Avg', mode='lines',
                   line=dict(color='#f59e0b', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Growth rate bars
    colors = ['#10b981' if x >= 0 else '#ef4444' for x in df_resampled['growth_rate'].fillna(0)]
    fig.add_trace(
        go.Bar(x=df_resampled['date'], y=df_resampled['growth_rate'],
               name='Growth %', marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(title=f'Sales Trend Analysis ({period.capitalize()})', height=600)
    
    # Calculate insights
    total_revenue = df_resampled['revenue'].sum()
    avg_growth = df_resampled['growth_rate'].mean()
    best_period = df_resampled.loc[df_resampled['revenue'].idxmax()]
    worst_period = df_resampled.loc[df_resampled['revenue'].idxmin()]
    recent_trend = df_resampled['growth_rate'].tail(3).mean()
    
    insights = {
        "title": f"Sales Trend Analysis ({period.capitalize()})",
        "key_findings": [
            f"💰 Total Revenue: ${total_revenue:,.0f}",
            f"📈 Average Growth Rate: {avg_growth:.1f}% per {period}",
            f"🔥 Best Period: {best_period['date'].strftime('%Y-%m-%d')} (${best_period['revenue']:,.0f})",
            f"📉 Recent Trend: {'Upward' if recent_trend > 0 else 'Downward'} ({recent_trend:.1f}%)",
        ],
        "recommendations": []
    }
    
    if recent_trend < -5:
        insights["recommendations"].append("⚠️ **Alert**: Recent sales are declining. Investigate causes and consider promotional campaigns.")
    if recent_trend > 10:
        insights["recommendations"].append("🚀 **Opportunity**: Strong growth momentum! Consider scaling marketing spend.")
    
    insights["recommendations"].append(f"📅 Focus marketing efforts around periods similar to {best_period['date'].strftime('%B')} when sales peak.")
    
    return {
        "charts": [fig.to_json()],
        "insights": insights,
        "data_preview": df_resampled.tail(10).to_dict('records')
    }


def product_performance_analysis(conn, metric='revenue'):
    """
    Pareto analysis of products - identify the vital few vs trivial many.
    """
    query = """
    SELECT 
        p.id as product_id,
        p.name as product_name,
        p.category,
        p.price,
        SUM(oi.quantity) as total_quantity,
        COUNT(DISTINCT o.id) as total_orders,
        SUM(oi.amount) as total_revenue
    FROM products p
    LEFT JOIN order_items oi ON p.id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.id AND o.status != 'Returned'
    GROUP BY p.id, p.name, p.category, p.price
    ORDER BY total_revenue DESC
    """
    
    df = pd.read_sql_query(query, conn)
    
    metric_col = {'revenue': 'total_revenue', 'quantity': 'total_quantity', 'orders': 'total_orders'}[metric]
    
    # Calculate cumulative percentage for Pareto
    df = df.sort_values(metric_col, ascending=False)
    df['cumulative'] = df[metric_col].cumsum()
    df['cumulative_pct'] = df['cumulative'] / df[metric_col].sum() * 100
    
    # Identify Pareto segments
    df['pareto_class'] = df['cumulative_pct'].apply(
        lambda x: 'Top 20% (Stars)' if x <= 20 else ('Middle 30%' if x <= 50 else 'Bottom 50%')
    )
    
    # Chart 1: Pareto chart
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig1.add_trace(
        go.Bar(x=df['product_name'], y=df[metric_col], name=metric.capitalize(),
               marker_color='#3b82f6'),
        secondary_y=False
    )
    
    fig1.add_trace(
        go.Scatter(x=df['product_name'], y=df['cumulative_pct'], name='Cumulative %',
                   mode='lines+markers', line=dict(color='#f59e0b', width=2)),
        secondary_y=True
    )
    
    # Add 80% line
    fig1.add_hline(y=80, line_dash="dash", line_color="red", 
                   annotation_text="80% of value", secondary_y=True)
    
    fig1.update_layout(title=f'Product Pareto Analysis ({metric.capitalize()})')
    fig1.update_xaxes(tickangle=45)
    fig1.update_yaxes(title_text=metric.capitalize(), secondary_y=False)
    fig1.update_yaxes(title_text="Cumulative %", secondary_y=True)
    
    # Chart 2: Category performance
    category_df = df.groupby('category').agg({
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_orders': 'sum'
    }).reset_index()
    
    fig2 = px.bar(
        category_df, 
        x='category', 
        y='total_revenue',
        color='category',
        title='Revenue by Category',
        text_auto='.2s'
    )
    
    # Calculate insights
    top_20_products = df[df['cumulative_pct'] <= 20]
    bottom_50_products = df[df['cumulative_pct'] > 50]
    
    insights = {
        "title": f"Product Performance Analysis (by {metric.capitalize()})",
        "key_findings": [
            f"📦 {len(top_20_products)} products generate 20% of {metric} (Stars)",
            f"🌟 Top performer: {df.iloc[0]['product_name']} (${df.iloc[0]['total_revenue']:,.0f})",
            f"📉 {len(bottom_50_products)} products in bottom 50% - consider discontinuing",
            f"🏷️ Best category: {category_df.loc[category_df['total_revenue'].idxmax(), 'category']}",
        ],
        "recommendations": [
            f"⭐ **Double down on Stars**: {', '.join(top_20_products['product_name'].head(3).tolist())}",
            "📦 **Review slow movers**: Consider bundling or discontinuing bottom performers",
            "💡 **Cross-sell opportunity**: Bundle top products with related items",
        ]
    }
    
    return {
        "charts": [fig1.to_json(), fig2.to_json()],
        "insights": insights,
        "data_preview": df[['product_name', 'category', 'total_revenue', 'total_quantity', 'pareto_class']].head(10).to_dict('records')
    }


def correlation_analysis(conn):
    """
    Analyze correlations between key business metrics.
    """
    query = """
    SELECT 
        p.price,
        p.category,
        oi.quantity,
        oi.amount,
        o.date
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    JOIN orders o ON oi.order_id = o.id
    WHERE o.status != 'Returned'
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Calculate correlation matrix for numeric columns
    numeric_df = df[['price', 'quantity', 'amount']].copy()
    corr_matrix = numeric_df.corr()
    
    # Chart 1: Correlation heatmap
    fig1 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix',
        labels=dict(color="Correlation")
    )
    
    # Chart 2: Price vs Quantity scatter
    fig2 = px.scatter(
        df, x='price', y='quantity',
        color='category',
        title='Price vs Quantity Sold',
        trendline='ols',
        labels={'price': 'Product Price ($)', 'quantity': 'Quantity per Order'}
    )
    
    # Calculate average quantity by price bracket
    df['price_bracket'] = pd.cut(df['price'], bins=[0, 25, 50, 75, 100], labels=['$0-25', '$25-50', '$50-75', '$75+'])
    price_analysis = df.groupby('price_bracket')['quantity'].mean().reset_index()
    
    insights = {
        "title": "Correlation & Relationship Analysis",
        "key_findings": [
            f"📊 Price-Quantity correlation: {corr_matrix.loc['price', 'quantity']:.2f} ({'Negative' if corr_matrix.loc['price', 'quantity'] < 0 else 'Positive'})",
            f"💵 Higher priced items tend to sell {'fewer' if corr_matrix.loc['price', 'quantity'] < 0 else 'more'} units per order",
            f"🎯 Sweet spot: ${df.groupby('price_bracket')['amount'].sum().idxmax()} price range generates most revenue",
        ],
        "recommendations": [
            "💡 Consider price elasticity when setting prices",
            "📦 Bundle high-price items to increase perceived value",
            "🎯 Test pricing in the optimal range for new products",
        ]
    }
    
    return {
        "charts": [fig1.to_json(), fig2.to_json()],
        "insights": insights,
        "data_preview": price_analysis.to_dict('records')
    }


def pca_analysis(conn, target='customers'):
    """
    Principal Component Analysis to find hidden patterns.
    """
    if target == 'customers':
        query = """
        SELECT 
            c.id as customer_id,
            c.name,
            c.country,
            COUNT(DISTINCT o.id) as order_count,
            SUM(oi.amount) as total_spent,
            AVG(oi.amount) as avg_order_value,
            SUM(oi.quantity) as total_items,
            COUNT(DISTINCT p.category) as categories_bought
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        WHERE o.status != 'Returned'
        GROUP BY c.id, c.name, c.country
        """
        id_col = 'customer_id'
        label_col = 'name'
        color_col = 'country'
    else:  # products
        query = """
        SELECT 
            p.id as product_id,
            p.name,
            p.category,
            p.price,
            SUM(oi.quantity) as total_sold,
            COUNT(DISTINCT o.customer_id) as unique_buyers,
            SUM(oi.amount) as total_revenue,
            COUNT(DISTINCT o.id) as times_ordered
        FROM products p
        LEFT JOIN order_items oi ON p.id = oi.product_id
        LEFT JOIN orders o ON oi.order_id = o.id AND o.status != 'Returned'
        GROUP BY p.id, p.name, p.category, p.price
        """
        id_col = 'product_id'
        label_col = 'name'
        color_col = 'category'
    
    df = pd.read_sql_query(query, conn)
    
    # Select numeric features for PCA
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != id_col]
    
    # Standardize features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols].fillna(0))
    
    # Perform PCA
    pca = PCA(n_components=min(3, len(numeric_cols)))
    pca_result = pca.fit_transform(scaled_data)
    
    # Add PCA results to dataframe
    df['PC1'] = pca_result[:, 0]
    df['PC2'] = pca_result[:, 1]
    if pca_result.shape[1] > 2:
        df['PC3'] = pca_result[:, 2]
    
    # Chart 1: PCA scatter plot
    fig1 = px.scatter(
        df, x='PC1', y='PC2',
        color=color_col,
        hover_data=[label_col] + numeric_cols[:3],
        title=f'PCA Analysis - {target.capitalize()} Patterns',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'}
    )
    
    # Chart 2: Explained variance
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'Variance Explained': pca.explained_variance_ratio_ * 100,
        'Cumulative': np.cumsum(pca.explained_variance_ratio_) * 100
    })
    
    fig2 = px.bar(
        variance_df, x='Component', y='Variance Explained',
        title='Variance Explained by Principal Components',
        text_auto='.1f'
    )
    
    # Feature importance
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=numeric_cols
    )
    
    top_features_pc1 = feature_importance['PC1'].abs().nlargest(3).index.tolist()
    
    insights = {
        "title": f"PCA Analysis - {target.capitalize()}",
        "key_findings": [
            f"📊 PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance",
            f"📈 First 2 components explain {sum(pca.explained_variance_ratio_[:2])*100:.1f}% of total variance",
            f"🔑 Most important factors: {', '.join(top_features_pc1)}",
        ],
        "recommendations": [
            f"💡 Focus on **{top_features_pc1[0]}** - it explains most {target} differences",
            "🎯 Clusters in the PCA plot suggest natural groupings you can target",
            "📉 Outliers in the plot may indicate unusual behavior worth investigating",
        ]
    }
    
    return {
        "charts": [fig1.to_json(), fig2.to_json()],
        "insights": insights,
        "data_preview": df[[label_col, color_col, 'PC1', 'PC2'] + numeric_cols[:3]].head(10).to_dict('records'),
        "feature_importance": feature_importance.to_dict()
    }


def customer_clustering(conn, n_clusters=4):
    """
    K-means clustering of customers based on behavior.
    """
    query = """
    SELECT 
        c.id as customer_id,
        c.name,
        c.country,
        COUNT(DISTINCT o.id) as order_count,
        SUM(oi.amount) as total_spent,
        AVG(oi.amount) as avg_order_value,
        SUM(oi.quantity) as total_items,
        julianday('now') - julianday(MAX(o.date)) as days_since_last_order
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.status != 'Returned'
    GROUP BY c.id, c.name, c.country
    """
    
    df = pd.read_sql_query(query, conn)
    
    # Features for clustering
    features = ['order_count', 'total_spent', 'avg_order_value', 'total_items', 'days_since_last_order']
    
    # Standardize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features].fillna(0))
    
    # K-means clustering
    # Suppress joblib warning on Windows by setting n_init explicitly
    import os
    if 'LOKY_MAX_CPU_COUNT' not in os.environ:
        os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to avoid warning
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Name clusters based on characteristics
    cluster_profiles = df.groupby('cluster')[features].mean()
    
    def name_cluster(row):
        if row['total_spent'] > cluster_profiles['total_spent'].median() and row['order_count'] > cluster_profiles['order_count'].median():
            return 'High-Value Regulars'
        elif row['total_spent'] > cluster_profiles['total_spent'].median():
            return 'Big Spenders'
        elif row['days_since_last_order'] > cluster_profiles['days_since_last_order'].median():
            return 'Dormant'
        elif row['order_count'] < cluster_profiles['order_count'].median():
            return 'Newcomers'
        else:
            return 'Average Customers'
    
    # Get base names for each cluster
    base_names = cluster_profiles.apply(name_cluster, axis=1).to_dict()
    
    # Ensure each cluster gets a unique name (add number suffix if duplicates exist)
    name_counts = {}
    cluster_names = {}
    for cluster_id, base_name in base_names.items():
        if base_name not in name_counts:
            name_counts[base_name] = 0
        name_counts[base_name] += 1
        
        if name_counts[base_name] == 1:
            cluster_names[cluster_id] = base_name
        else:
            # Add suffix for duplicate names
            cluster_names[cluster_id] = f"{base_name} {name_counts[base_name]}"
    
    df['cluster_name'] = df['cluster'].map(cluster_names)
    
    # Fill any NaN cluster names (shouldn't happen, but safety check)
    if df['cluster_name'].isna().any():
        df['cluster_name'] = df['cluster_name'].fillna('Unknown')
    
    # Get actual number of distinct segments (unique cluster names)
    actual_segments = df['cluster_name'].nunique()
    
    # Chart 1: Cluster visualization (2D using top 2 features)
    fig1 = px.scatter(
        df, x='total_spent', y='order_count',
        color='cluster_name',
        size='avg_order_value',
        hover_data=['name', 'country'],
        title='Customer Clusters',
        labels={'total_spent': 'Total Spent ($)', 'order_count': 'Number of Orders'}
    )
    
    # Chart 2: Cluster characteristics radar/bar
    cluster_summary = df.groupby('cluster_name').agg({
        'customer_id': 'count',
        'total_spent': 'mean',
        'order_count': 'mean',
        'avg_order_value': 'mean'
    }).round(2).reset_index()
    cluster_summary.columns = ['Cluster', 'Count', 'Avg Total Spent', 'Avg Orders', 'Avg Order Value']
    
    fig2 = px.bar(
        cluster_summary, x='Cluster', y='Count',
        color='Avg Total Spent',
        title='Customers per Cluster',
        text_auto=True
    )
    
    # Build insights with safe access to cluster_summary
    # Use actual number of distinct segments (unique cluster names) instead of n_clusters
    key_findings = [
        f"👥 Identified {actual_segments} distinct customer segments",
    ]
    
    if not cluster_summary.empty:
        try:
            max_spent_idx = cluster_summary['Avg Total Spent'].idxmax()
            highest_value_cluster = cluster_summary.loc[max_spent_idx, 'Cluster']
            key_findings.append(f"⭐ '{highest_value_cluster}' segment has highest value")
        except Exception:
            pass
        
        try:
            max_count_idx = cluster_summary['Count'].idxmax()
            largest_cluster = cluster_summary.loc[max_count_idx, 'Cluster']
            largest_count = cluster_summary['Count'].max()
            key_findings.append(f"📊 Largest segment: {largest_cluster} ({largest_count} customers)")
        except Exception:
            pass
    
    insights = {
        "title": "Customer Clustering Analysis",
        "key_findings": key_findings,
        "recommendations": [
            "🎯 Create targeted campaigns for each segment",
            "💎 Focus retention efforts on high-value clusters",
            "🔄 Design win-back campaigns for dormant customers",
            "📧 Personalize email content by cluster",
        ],
        "cluster_profiles": cluster_summary.to_dict('records') if not cluster_summary.empty else []
    }
    
    return {
        "charts": [fig1.to_json(), fig2.to_json()],
        "insights": insights,
        "data_preview": df[['name', 'cluster_name', 'total_spent', 'order_count', 'country']].head(10).to_dict('records')
    }


def cohort_analysis(conn):
    """
    Customer retention cohort analysis.
    """
    query = """
    SELECT 
        c.id as customer_id,
        c.signup_date,
        o.date as order_date,
        SUM(oi.amount) as order_amount
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.status != 'Returned'
    GROUP BY c.id, c.signup_date, o.date
    """
    
    df = pd.read_sql_query(query, conn)
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Create cohort month
    df['cohort_month'] = df['signup_date'].dt.to_period('M')
    df['order_month'] = df['order_date'].dt.to_period('M')
    
    # Calculate months since signup
    df['months_since_signup'] = (df['order_month'] - df['cohort_month']).apply(lambda x: x.n if pd.notna(x) else 0)
    
    # Create cohort table
    cohort_data = df.groupby(['cohort_month', 'months_since_signup'])['customer_id'].nunique().unstack(fill_value=0)
    
    # Calculate retention rates
    cohort_sizes = cohort_data.iloc[:, 0]
    retention = cohort_data.divide(cohort_sizes, axis=0) * 100
    
    # Chart: Retention heatmap
    fig = px.imshow(
        retention.iloc[:6, :6],  # Show first 6 cohorts and 6 months
        text_auto='.0f',
        color_continuous_scale='Blues',
        title='Customer Retention by Cohort (%)',
        labels=dict(x='Months Since Signup', y='Signup Cohort', color='Retention %')
    )
    
    # Calculate insights
    avg_retention_m1 = retention.iloc[:, 1].mean() if retention.shape[1] > 1 else 0
    avg_retention_m3 = retention.iloc[:, 3].mean() if retention.shape[1] > 3 else 0
    
    insights = {
        "title": "Cohort Retention Analysis",
        "key_findings": [
            f"📊 Average Month-1 retention: {avg_retention_m1:.1f}%",
            f"📉 Average Month-3 retention: {avg_retention_m3:.1f}%",
            f"👥 {len(cohort_sizes)} monthly cohorts analyzed",
        ],
        "recommendations": [
            "🎯 Focus on early engagement - biggest drop is in month 1",
            "📧 Set up automated re-engagement emails at day 30, 60, 90",
            "💡 Analyze what top-retained cohorts have in common",
        ]
    }
    
    return {
        "charts": [fig.to_json()],
        "insights": insights,
        "data_preview": retention.head().to_dict()
    }


def geographic_analysis(conn):
    """
    Analyze performance by geographic region.
    """
    query = """
    SELECT 
        c.country,
        COUNT(DISTINCT c.id) as customer_count,
        COUNT(DISTINCT o.id) as order_count,
        SUM(oi.amount) as total_revenue,
        AVG(oi.amount) as avg_order_value,
        SUM(oi.quantity) as total_units
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.status != 'Returned'
    GROUP BY c.country
    ORDER BY total_revenue DESC
    """
    
    df = pd.read_sql_query(query, conn)
    
    df['revenue_per_customer'] = df['total_revenue'] / df['customer_count']
    df['orders_per_customer'] = df['order_count'] / df['customer_count']
    
    # Chart 1: Revenue by country
    fig1 = px.bar(
        df, x='country', y='total_revenue',
        color='total_revenue',
        title='Revenue by Country',
        text_auto='.2s',
        color_continuous_scale='Blues'
    )
    
    # Chart 2: Customer value comparison
    fig2 = px.scatter(
        df, x='customer_count', y='revenue_per_customer',
        size='total_revenue',
        color='country',
        title='Market Opportunity (Customers vs Revenue per Customer)',
        labels={'customer_count': 'Number of Customers', 'revenue_per_customer': 'Revenue per Customer ($)'}
    )
    
    top_market = df.iloc[0]
    highest_value = df.loc[df['revenue_per_customer'].idxmax()]
    
    insights = {
        "title": "Geographic Performance Analysis",
        "key_findings": [
            f"🌍 Top market: {top_market['country']} (${top_market['total_revenue']:,.0f} revenue)",
            f"💎 Highest value customers: {highest_value['country']} (${highest_value['revenue_per_customer']:,.0f}/customer)",
            f"📊 {len(df)} countries with active customers",
        ],
        "recommendations": [
            f"🚀 **Scale**: Invest more in {top_market['country']} - proven market",
            f"💡 **Learn**: Study why {highest_value['country']} customers spend more",
            "🎯 Consider localized marketing for top 3 markets",
        ]
    }
    
    return {
        "charts": [fig1.to_json(), fig2.to_json()],
        "insights": insights,
        "data_preview": df.to_dict('records')
    }


# =============================================================================
# MAIN DISPATCHER
# =============================================================================

def run_analysis(analysis_name, conn, **kwargs):
    """
    Dispatch to the appropriate analysis function.
    """
    analysis_map = {
        'rfm_analysis': rfm_analysis,
        'sales_trend_analysis': sales_trend_analysis,
        'product_performance_analysis': product_performance_analysis,
        'correlation_analysis': correlation_analysis,
        'pca_analysis': pca_analysis,
        'customer_clustering': customer_clustering,
        'cohort_analysis': cohort_analysis,
        'geographic_analysis': geographic_analysis,
    }
    
    if analysis_name not in analysis_map:
        return None
    
    func = analysis_map[analysis_name]
    
    # Filter kwargs to only valid parameters for each function
    import inspect
    valid_params = inspect.signature(func).parameters.keys()
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params and k != 'conn'}
    
    return func(conn, **filtered_kwargs)

