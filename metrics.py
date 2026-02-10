import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load your data
df = pd.read_csv('skygeni_sales_data.csv')
df['created_date'] = pd.to_datetime(df['created_date'])
df['closed_date'] = pd.to_datetime(df['closed_date'])

# ============================================================
# METRIC 1: Pipeline Momentum Index (PMI)
# ============================================================
# A predictive metric that measures the "velocity and quality" momentum of deals
# Combines: deal size acceleration, stage progression speed, and win probability momentum

def calculate_pipeline_momentum(df):
    """
    Calculate Pipeline Momentum Index (0-100)
    Higher scores indicate deals with accelerating momentum toward closing
    
    Components:
    1. Deal Size Momentum: Recent deal size trend vs historical
    2. Stage Progression Speed: How quickly deals move between stages
    3. Win Probability Acceleration: Improving win likelihood over time
    4. Engagement Frequency: Activity level based on deal age
    """
    
    # Create a copy to avoid modifying original
    momentum_df = df.copy()
    
    # 1. Calculate Deal Size Momentum (compared to industry/region avg)
    industry_avg = momentum_df.groupby('industry')['deal_amount'].transform('mean')
    region_avg = momentum_df.groupby('region')['deal_amount'].transform('mean')
    momentum_df['size_momentum'] = ((momentum_df['deal_amount'] - industry_avg) / industry_avg * 50 + 50)
    momentum_df['size_momentum'] = momentum_df['size_momentum'].clip(0, 100)
    
    # 2. Calculate Stage Progression Score
    # Map deal stages to numerical progression values
    stage_map = {
        'Qualified': 10,
        'Demo': 25,
        'Proposal': 50,
        'Negotiation': 75,
        'Closed': 100
    }
    momentum_df['stage_score'] = momentum_df['deal_stage'].map(stage_map).fillna(0)
    
    # Calculate progression speed (days per stage point)
    momentum_df['deal_age_days'] = (momentum_df['closed_date'] - momentum_df['created_date']).dt.days
    momentum_df['progression_speed'] = np.where(
        momentum_df['deal_age_days'] > 0,
        momentum_df['stage_score'] / momentum_df['deal_age_days'],
        0
    )
    
    # Normalize progression speed (0-100)
    if momentum_df['progression_speed'].max() > 0:
        momentum_df['progression_momentum'] = (
            momentum_df['progression_speed'] / momentum_df['progression_speed'].max() * 100
        )
    else:
        momentum_df['progression_momentum'] = 0
    
    # 3. Calculate Win Probability Momentum
    # Calculate historical win rates by source/industry
    win_rates = momentum_df.groupby(['lead_source', 'industry'])['outcome'].apply(
        lambda x: (x == 'Won').mean()
    ).reset_index(name='historical_win_rate')
    
    momentum_df = momentum_df.merge(win_rates, on=['lead_source', 'industry'], how='left')
    momentum_df['historical_win_rate'] = momentum_df['historical_win_rate'].fillna(0.5)
    
    # Adjust for recent performance (last 90 days trend)
    current_date = momentum_df['created_date'].max()
    recent_date = current_date - pd.Timedelta(days=90)
    recent_deals = momentum_df[momentum_df['created_date'] > recent_date]
    
    if len(recent_deals) > 0:
        recent_win_rates = recent_deals.groupby(['lead_source', 'industry'])['outcome'].apply(
            lambda x: (x == 'Won').mean()
        ).reset_index(name='recent_win_rate')
        
        momentum_df = momentum_df.merge(recent_win_rates, on=['lead_source', 'industry'], how='left')
        momentum_df['recent_win_rate'] = momentum_df['recent_win_rate'].fillna(momentum_df['historical_win_rate'])
        
        # Win momentum = improvement in win rate
        momentum_df['win_momentum'] = (
            (momentum_df['recent_win_rate'] - momentum_df['historical_win_rate']) * 200 + 50
        ).clip(0, 100)
    else:
        momentum_df['win_momentum'] = 50
    
    # 4. Calculate Engagement Frequency Score
    # Shorter cycle deals get higher momentum scores
    momentum_df['cycle_efficiency'] = np.where(
        momentum_df['sales_cycle_days'] > 0,
        np.exp(-momentum_df['sales_cycle_days'] / 90) * 100,  # Exponential decay based on 90-day benchmark
        100
    )
    
    # ============================================================
    # FINAL PMI CALCULATION
    # Weighted average of all momentum components
    # ============================================================
    weights = {
        'size_momentum': 0.25,      # Deal size growing vs peers
        'progression_momentum': 0.30,  # Stage progression speed
        'win_momentum': 0.25,       # Improving win probability
        'cycle_efficiency': 0.20    # Engagement frequency
    }
    
    momentum_df['pipeline_momentum_index'] = (
        momentum_df['size_momentum'] * weights['size_momentum'] +
        momentum_df['progression_momentum'] * weights['progression_momentum'] +
        momentum_df['win_momentum'] * weights['win_momentum'] +
        momentum_df['cycle_efficiency'] * weights['cycle_efficiency']
    )
    
    # Scale to 0-100
    momentum_df['pipeline_momentum_index'] = momentum_df['pipeline_momentum_index'].clip(0, 100)
    
    return momentum_df[['deal_id', 'pipeline_momentum_index', 'size_momentum', 
                       'progression_momentum', 'win_momentum', 'cycle_efficiency']]

# ============================================================
# METRIC 2: Strategic Account Value Score (SAVS)
# ============================================================
# Measures the long-term strategic value of accounts, not just current deal value
# Focuses on: industry influence, expansion potential, partnership value, and competitive defense

def calculate_strategic_account_value(df):
    """
    Calculate Strategic Account Value Score (0-100)
    Identifies accounts with high strategic importance beyond immediate revenue
    
    Components:
    1. Industry Influence Multiplier: Deal's impact on industry presence
    2. Expansion Potential Score: Likelihood of cross/up-sell
    3. Partnership Value: Referral and case study potential
    4. Competitive Defense Value: Preventing competitor wins
    """
    
    strategic_df = df.copy()
    
    # 1. Industry Influence Multiplier
    # Some industries are more strategic than others
    industry_strategy_map = {
        'FinTech': 1.8,      # High strategic value (digital transformation)
        'HealthTech': 1.6,    # Strategic (regulated, growing)
        'SaaS': 1.4,         # Moderate strategic value
        'Ecommerce': 1.3,     # Moderate
        'EdTech': 1.5        # Strategic (education is key market)
    }
    
    strategic_df['industry_influence'] = strategic_df['industry'].map(
        lambda x: industry_strategy_map.get(x, 1.2)
    )
    
    # Regional strategic importance
    region_strategy_map = {
        'North America': 1.5,
        'Europe': 1.4,
        'APAC': 1.6,      # High growth potential
        'India': 1.3
    }
    
    strategic_df['region_multiplier'] = strategic_df['region'].map(
        lambda x: region_strategy_map.get(x, 1.0)
    )
    
    # 2. Expansion Potential Score
    # Based on product type and deal patterns
    expansion_potential = {
        'Core': 1.0,        # Base product - moderate expansion
        'Enterprise': 2.0,   # High expansion potential
        'Pro': 1.5          # Professional - good upsell
    }
    
    strategic_df['expansion_score'] = strategic_df['product_type'].map(
        lambda x: expansion_potential.get(x, 1.0)
    )
    
    # Add multiplier for large deals (more room to expand)
    strategic_df['deal_size_multiplier'] = np.where(
        strategic_df['deal_amount'] > strategic_df['deal_amount'].median(),
        1.5,
        1.0
    )
    
    # 3. Partnership Value Score
    # Referral sources and successful deals create partnership value
    partnership_value = {
        'Referral': 2.0,    # Already referring others
        'Partner': 1.8,     # Partner channel - strategic
        'Inbound': 1.2,     # Good for case studies
        'Outbound': 1.0     # Standard
    }
    
    strategic_df['partnership_score'] = strategic_df['lead_source'].map(
        lambda x: partnership_value.get(x, 1.0)
    )
    
    # Won deals have higher partnership potential (reference accounts)
    strategic_df['outcome_multiplier'] = np.where(
        strategic_df['outcome'] == 'Won',
        1.8,  # Successful implementations are valuable references
        1.0
    )
    
    # 4. Competitive Defense Value
    # Calculate how "competitive" this account is based on deal characteristics
    strategic_df['competition_risk'] = np.where(
        strategic_df['sales_cycle_days'] > 60,
        1.5,  # Long cycles suggest competitive situations
        1.0
    )
    
    # Large deals in strategic industries have high defense value
    strategic_df['defense_value'] = np.where(
        (strategic_df['deal_amount'] > df['deal_amount'].quantile(0.75)) &
        (strategic_df['industry'].isin(['FinTech', 'HealthTech'])),
        2.0,
        1.0
    )
    
    # ============================================================
    # CALCULATE STRATEGIC ACCOUNT VALUE SCORE (SAVS)
    # ============================================================
    
    # Base score from deal amount (normalized 0-100)
    strategic_df['deal_amount_score'] = (
        (strategic_df['deal_amount'] - strategic_df['deal_amount'].min()) /
        (strategic_df['deal_amount'].max() - strategic_df['deal_amount'].min()) * 100
    ).fillna(50)
    
    # Apply all strategic multipliers
    strategic_df['strategic_multiplier'] = (
        strategic_df['industry_influence'] *
        strategic_df['region_multiplier'] *
        strategic_df['expansion_score'] *
        strategic_df['deal_size_multiplier'] *
        strategic_df['partnership_score'] *
        strategic_df['outcome_multiplier'] *
        strategic_df['defense_value']
    )
    
    # Final SAVS calculation
    strategic_df['strategic_account_value_score'] = (
        strategic_df['deal_amount_score'] *
        strategic_df['strategic_multiplier'] *
        0.01  # Scale down to reasonable range
    ).clip(0, 100)
    
    return strategic_df[['deal_id', 'strategic_account_value_score', 
                         'industry_influence', 'region_multiplier', 
                         'expansion_score', 'partnership_score', 
                         'defense_value', 'strategic_multiplier']]

# ============================================================
# RUN THE ANALYSIS
# ============================================================

print("=" * 60)
print("SKYGENI CUSTOM METRICS ANALYSIS")
print("=" * 60)

# Calculate Pipeline Momentum Index
print("\n1. CALCULATING PIPELINE MOMENTUM INDEX (PMI)...")
pmi_results = calculate_pipeline_momentum(df)

print(f"\nPMI Statistics:")
print(f"Average PMI: {pmi_results['pipeline_momentum_index'].mean():.1f}")
print(f"Top 10% PMI Threshold: {pmi_results['pipeline_momentum_index'].quantile(0.9):.1f}")
print(f"Bottom 10% PMI Threshold: {pmi_results['pipeline_momentum_index'].quantile(0.1):.1f}")

# Calculate Strategic Account Value Score
print("\n\n2. CALCULATING STRATEGIC ACCOUNT VALUE SCORE (SAVS)...")
savs_results = calculate_strategic_account_value(df)

print(f"\nSAVS Statistics:")
print(f"Average SAVS: {savs_results['strategic_account_value_score'].mean():.1f}")
print(f"Top Strategic Accounts (>80 SAVS): {len(savs_results[savs_results['strategic_account_value_score'] > 80])} deals")
print(f"Most Strategic Industry: {df.groupby('industry')['deal_amount'].sum().idxmax()}")

# ============================================================
# GENERATE ACTIONABLE INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("ACTIONABLE INSIGHTS")
print("=" * 60)

# Merge results for comprehensive analysis
combined_results = pd.merge(
    pmi_results, 
    savs_results, 
    on='deal_id', 
    how='inner'
)

# Identify high-potential deals (high on both metrics)
combined_results['high_potential'] = (
    (combined_results['pipeline_momentum_index'] > 70) & 
    (combined_results['strategic_account_value_score'] > 70)
)

high_potential_deals = combined_results[combined_results['high_potential']]
print(f"\n🔍 HIGH-POTENTIAL DEALS IDENTIFIED: {len(high_potential_deals)}")
print("   (High Pipeline Momentum AND High Strategic Value)")

if len(high_potential_deals) > 0:
    print("\nSample High-Potential Deals:")
    print(high_potential_deals[['deal_id', 'pipeline_momentum_index', 'strategic_account_value_score']].head())

# Identify deals needing attention
attention_needed = combined_results[
    (combined_results['pipeline_momentum_index'] < 30) & 
    (combined_results['strategic_account_value_score'] > 60)
]

print(f"\n⚠️  DEALS NEEDING ATTENTION: {len(attention_needed)}")
print("   (High Strategic Value but Low Pipeline Momentum)")

# ============================================================
# VISUALIZATION CODE (Optional - requires matplotlib)
# ============================================================

def generate_visualization_code():
    """Generate Python code for visualizing these metrics"""
    
    #viz_code = 
import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SkyGeni Custom Metrics Dashboard', fontsize=16, fontweight='bold')

# 1. PMI Distribution
axes[0, 0].hist(combined_results['pipeline_momentum_index'], bins=30, 
                color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(combined_results['pipeline_momentum_index'].mean(), 
                   color='red', linestyle='--', label=f"Mean: {combined_results['pipeline_momentum_index'].mean():.1f}")
axes[0, 0].set_title('Pipeline Momentum Index Distribution')
axes[0, 0].set_xlabel('PMI Score (0-100)')
axes[0, 0].set_ylabel('Number of Deals')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. SAVS by Industry
industry_savs = combined_results.merge(df[['deal_id', 'industry']], on='deal_id')
industry_avg = industry_savs.groupby('industry')['strategic_account_value_score'].mean().sort_values()
axes[0, 1].barh(range(len(industry_avg)), industry_avg.values, color='lightgreen')
axes[0, 1].set_yticks(range(len(industry_avg)))
axes[0, 1].set_yticklabels(industry_avg.index)
axes[0, 1].set_title('Strategic Account Value by Industry')
axes[0, 1].set_xlabel('Average SAVS Score')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Scatter Plot: PMI vs SAVS
scatter = axes[1, 0].scatter(combined_results['pipeline_momentum_index'], 
                            combined_results['strategic_account_value_score'],
                            c=combined_results['strategic_multiplier'], 
                            cmap='viridis', alpha=0.6, s=50)
axes[1, 0].set_title('PMI vs Strategic Account Value')
axes[1, 0].set_xlabel('Pipeline Momentum Index')
axes[1, 0].set_ylabel('Strategic Account Value Score')
plt.colorbar(scatter, ax=axes[1, 0], label='Strategic Multiplier')

# Add quadrants
axes[1, 0].axhline(50, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].axvline(50, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].text(25, 75, 'High Value\nLow Momentum', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
axes[1, 0].text(75, 75, 'High Value\nHigh Momentum', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
axes[1, 0].grid(True, alpha=0.3)

# 4. Top 10 Strategic Deals
top_strategic = combined_results.nlargest(10, 'strategic_account_value_score')
bars = axes[1, 1].bar(range(len(top_strategic)), top_strategic['strategic_account_value_score'],
                     color=['red' if x > 85 else 'orange' for x in top_strategic['strategic_account_value_score']])
axes[1, 1].set_title('Top 10 Most Strategic Deals')
axes[1, 1].set_xlabel('Deal Rank')
axes[1, 1].set_ylabel('SAVS Score')
axes[1, 1].set_xticks(range(len(top_strategic)))
axes[1, 1].set_xticklabels(top_strategic['deal_id'], rotation=45, ha='right')

# Add value labels on bars
for bar, value in zip(bars, top_strategic['strategic_account_value_score']):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

    
    #return viz_code

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("\n1. Monitor deals")
print("2. Monitor deals with PMI > 70 for acceleration opportunities")
print("3. Focus resources on deals with SAVS > 80 (high strategic value)")
print("4. Review deals in 'High Value / Low Momentum' quadrant weekly")
print("\nThese metrics help you:")
print("   • Prioritize sales efforts effectively")
print("   • Identify strategic accounts early")
print("   • Predict pipeline performance")
print("   • Allocate resources based on both value and momentum")

# Save results to CSV
combined_results.to_csv('skygeni_custom_metrics_results.csv', index=False)
print(f"\n✅ Results saved to: skygeni_custom_metrics_results.csv")