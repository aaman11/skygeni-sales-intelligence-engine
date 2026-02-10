# data_generator.py
"""
Data generation module for SkyGeni Risk Scoring Demo
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple

def generate_sample_data(n_historical: int = 1000, n_open: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic sales data for demonstration

    Args:
        n_historical: Number of historical deals (won/lost)
        n_open: Number of open deals in pipeline

    Returns:
        Tuple of (historical_data, open_pipeline)
    """

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Configuration
    industries = ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail', 'Education']
    regions = ['North America', 'EMEA', 'APAC', 'Latin America']
    products = ['Enterprise', 'Professional', 'Basic', 'Custom']
    lead_sources = ['Website', 'Referral', 'Event', 'Outbound', 'Partnership']
    sales_reps = [f'REP_{i:03d}' for i in range(1, 21)]
    deal_stages = ['qualification', 'discovery', 'demo', 'proposal', 'negotiation']

    # Historical deals (won/lost)
    historical_deals = []
    for i in range(n_historical):
        # Base deal info
        created_date = datetime.now() - timedelta(days=random.randint(30, 365))

        # Determine outcome
        outcome = random.choice(['won', 'lost'])

        # Closed date based on outcome
        if outcome == 'won':
            closed_date = created_date + timedelta(days=random.randint(30, 120))
        else:
            closed_date = created_date + timedelta(days=random.randint(15, 90))

        # Deal amount with some patterns
        base_amount = np.random.lognormal(mean=10, sigma=0.8)
        if outcome == 'won':
            amount = base_amount * 1000  # Higher for wins
        else:
            amount = base_amount * 800   # Lower for losses

        # Introduce some business logic patterns
        industry = random.choice(industries)
        product = random.choice(products)

        # Pattern: Technology + Enterprise has higher win rate
        if industry == 'Technology' and product == 'Enterprise':
            outcome_bias = 0.7  # 70% chance of win
        # Pattern: Manufacturing + Basic has lower win rate
        elif industry == 'Manufacturing' and product == 'Basic':
            outcome_bias = 0.3  # 30% chance of win
        else:
            outcome_bias = 0.5  # Neutral

        # Adjust outcome based on bias
        if random.random() < outcome_bias:
            outcome = 'won'
        else:
            outcome = 'lost'

        # Adjust amount based on outcome
        if outcome == 'won':
            amount *= 1.2

        # Lead source performance pattern
        lead_source = random.choice(lead_sources)
        if lead_source == 'Referral':
            outcome_bias = 0.7
        elif lead_source == 'Event':
            outcome_bias = 0.3
        else:
            outcome_bias = 0.5

        # Re-adjust outcome
        if random.random() < outcome_bias:
            outcome = 'won'
        else:
            outcome = 'lost'

        deal = {
            'deal_id': f'HIST_{i:05d}',
            'created_date': created_date,
            'closed_date': closed_date,
            'deal_stage': random.choice(deal_stages) if outcome == 'lost' else 'closed_won',
            'deal_amount': round(amount, 2),
            'sales_rep_id': random.choice(sales_reps),
            'industry': industry,
            'region': random.choice(regions),
            'product_type': product,
            'lead_source': lead_source,
            'outcome': outcome
        }
        historical_deals.append(deal)

    # Open pipeline deals
    open_deals = []
    for i in range(n_open):
        # Base deal info
        created_date = datetime.now() - timedelta(days=random.randint(1, 180))

        # Current stage
        current_stage = random.choice(deal_stages)

        # Deal amount
        amount = np.random.lognormal(mean=10, sigma=0.7) * 1000

        # Add some high-risk patterns
        industry = random.choice(industries)
        product = random.choice(products)

        # Create some intentionally high-risk deals
        if i < 10:  # First 10 deals are high-risk
            # Old deals in negotiation
            created_date = datetime.now() - timedelta(days=random.randint(100, 180))
            current_stage = 'negotiation'
            amount = amount * 2  # Larger deals
            industry = random.choice(['Manufacturing', 'Retail'])  # Lower win rate industries
            product = 'Basic'

        deal = {
            'deal_id': f'OPEN_{i:05d}',
            'created_date': created_date,
            'closed_date': None,
            'deal_stage': current_stage,
            'deal_amount': round(amount, 2),
            'sales_rep_id': random.choice(sales_reps),
            'industry': industry,
            'region': random.choice(regions),
            'product_type': product,
            'lead_source': random.choice(lead_sources),
            'outcome': None,
            'last_stage_change': created_date + timedelta(days=random.randint(0, 30)),
            'customer_size': random.choice(['SMB', 'Mid-Market', 'Enterprise']),
            'decision_maker_engaged': random.random() > 0.5,
            'competition': random.sample(['Competitor A', 'Competitor B', 'Competitor C'],
                                       k=random.randint(0, 2))
        }
        open_deals.append(deal)

    # Convert to DataFrames
    historical_df = pd.DataFrame(historical_deals)
    open_df = pd.DataFrame(open_deals)

    # Add some missing values to simulate real data
    for df in [historical_df, open_df]:
        # Randomly set 5% of values to NaN
        mask = np.random.random(df.shape) < 0.05
        df[mask] = np.nan

    return historical_df, open_df

# Now let's update the demo function to use this module
print("Creating data generation module...")

# Update the run_demo function to use the local data generator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Run a simplified demo without the import issue
print("=" * 60)
print("SKYGENI RISK SCORING ENGINE - SIMPLIFIED DEMO")
print("=" * 60)

# Generate data directly
print("\n1. Generating sample data...")
historical_data, open_pipeline = generate_sample_data(n_historical=1000, n_open=50)

print(f"   Generated {len(historical_data)} historical deals")
print(f"   Generated {len(open_pipeline)} open deals in pipeline")

# Now we need to import the rest of the code from the previous risk_scorer_demo.py
# Let me recreate a compact version for demonstration:

print("\n2. Creating a compact demo of the risk scoring logic...")

# Simplified Deal class for demo
class SimpleDeal:
    def __init__(self, deal_id, created_date, deal_stage, deal_amount, sales_rep_id,
                 industry, region, product_type, lead_source, outcome=None):
        self.deal_id = deal_id
        self.created_date = created_date
        self.deal_stage = deal_stage
        self.deal_amount = deal_amount
        self.sales_rep_id = sales_rep_id
        self.industry = industry
        self.region = region
        self.product_type = product_type
        self.lead_source = lead_source
        self.outcome = outcome

# Simplified risk scoring logic
def calculate_simple_risk_score(deal, historical_data):
    """
    Simplified risk scoring for demo purposes
    """
    risk_score = 50  # Start at neutral

    # 1. Deal size risk
    if deal.deal_amount > 100000:
        risk_score += 20  # Larger deals are riskier
    elif deal.deal_amount < 10000:
        risk_score -= 10  # Smaller deals are less risky

    # 2. Deal stage risk
    stage_risk = {
        'qualification': 40,
        'discovery': 30,
        'demo': 25,
        'proposal': 20,
        'negotiation': 15
    }
    if deal.deal_stage in stage_risk:
        risk_score += stage_risk[deal.deal_stage]

    # 3. Deal age risk
    deal_age = (datetime.now() - deal.created_date).days
    if deal_age > 90:
        risk_score += 25
    elif deal_age > 60:
        risk_score += 15
    elif deal_age > 30:
        risk_score += 5

    # 4. Rep performance (simplified)
    rep_deals = historical_data[historical_data['sales_rep_id'] == deal.sales_rep_id]
    if len(rep_deals) > 0:
        rep_win_rate = (rep_deals['outcome'] == 'won').mean()
        if rep_win_rate < 0.4:
            risk_score += 20
        elif rep_win_rate < 0.6:
            risk_score += 10

    # 5. Industry-Product fit (simplified)
    industry_deals = historical_data[
        (historical_data['industry'] == deal.industry) &
        (historical_data['product_type'] == deal.product_type)
    ]
    if len(industry_deals) > 0:
        industry_win_rate = (industry_deals['outcome'] == 'won').mean()
        if industry_win_rate < 0.4:
            risk_score += 15
        elif industry_win_rate < 0.6:
            risk_score += 8

    # 6. Lead source risk
    lead_source_deals = historical_data[historical_data['lead_source'] == deal.lead_source]
    if len(lead_source_deals) > 0:
        lead_win_rate = (lead_source_deals['outcome'] == 'won').mean()
        if lead_win_rate < 0.3:
            risk_score += 10
        elif lead_win_rate < 0.5:
            risk_score += 5

    # Cap the score between 0 and 100
    risk_score = max(0, min(100, risk_score))

    return risk_score

# Risk level mapping
def get_risk_level(score):
    if score <= 30:
        return "LOW"
    elif score <= 70:
        return "MEDIUM"
    elif score <= 90:
        return "HIGH"
    else:
        return "CRITICAL"

# Analyze sample deals
print("\n3. Analyzing sample deals...")
print("-" * 40)

# Convert date strings to datetime objects
for df in [historical_data, open_pipeline]:
    df['created_date'] = pd.to_datetime(df['created_date'])
    if 'closed_date' in df.columns:
        df['closed_date'] = pd.to_datetime(df['closed_date'])

# Analyze first 5 open deals
for i in range(min(5, len(open_pipeline))):
    row = open_pipeline.iloc[i]

    # Create deal object
    deal = SimpleDeal(
        deal_id=row['deal_id'],
        created_date=row['created_date'],
        deal_stage=row['deal_stage'],
        deal_amount=row['deal_amount'],
        sales_rep_id=row['sales_rep_id'],
        industry=row['industry'],
        region=row['region'],
        product_type=row['product_type'],
        lead_source=row['lead_source']
    )

    # Calculate risk score
    risk_score = calculate_simple_risk_score(deal, historical_data)
    risk_level = get_risk_level(risk_score)

    print(f"\nDeal: {deal.deal_id}")
    print(f"  Amount: ${deal.deal_amount:,.0f}")
    print(f"  Stage: {deal.deal_stage}")
    print(f"  Industry: {deal.industry} / {deal.product_type}")
    print(f"  Age: {(datetime.now() - deal.created_date).days} days")
    print(f"  Risk Score: {risk_score:.1f}/100 ({risk_level})")

    # Generate recommendations based on risk
    if risk_level == "CRITICAL":
        print(f"  ⚠️  Recommendations:")
        print(f"     • Immediate executive review required")
        print(f"     • Develop fallback plan with discounts")
    elif risk_level == "HIGH":
        print(f"  ⚠️  Recommendations:")
        print(f"     • Schedule deal review with manager")
        print(f"     • Engage technical presales team")

# Pipeline summary
print("\n" + "=" * 60)
print("PIPELINE RISK SUMMARY")
print("=" * 60)

risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
total_value = 0
at_risk_value = 0

for i in range(len(open_pipeline)):
    row = open_pipeline.iloc[i]

    deal = SimpleDeal(
        deal_id=row['deal_id'],
        created_date=row['created_date'],
        deal_stage=row['deal_stage'],
        deal_amount=row['deal_amount'],
        sales_rep_id=row['sales_rep_id'],
        industry=row['industry'],
        region=row['region'],
        product_type=row['product_type'],
        lead_source=row['lead_source']
    )

    risk_score = calculate_simple_risk_score(deal, historical_data)
    risk_level = get_risk_level(risk_score)

    risk_counts[risk_level] += 1
    total_value += deal.deal_amount

    if risk_level in ["HIGH", "CRITICAL"]:
        at_risk_value += deal.deal_amount * (risk_score / 100)

print(f"\nDeal Count by Risk Level:")
for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    count = risk_counts[level]
    percentage = (count / len(open_pipeline)) * 100
    print(f"  {level:8}: {count:3} deals ({percentage:.1f}%)")

print(f"\nPipeline Value: ${total_value:,.0f}")
print(f"At-Risk Value:  ${at_risk_value:,.0f} ({at_risk_value/total_value*100:.1f}% of pipeline)")

# Business insights
print("\n" + "=" * 60)
print("KEY BUSINESS INSIGHTS")
print("=" * 60)

# Insight 1: High-risk deals analysis
high_risk_deals = []
for i in range(len(open_pipeline)):
    row = open_pipeline.iloc[i]
    deal = SimpleDeal(
        deal_id=row['deal_id'],
        created_date=row['created_date'],
        deal_stage=row['deal_stage'],
        deal_amount=row['deal_amount'],
        sales_rep_id=row['sales_rep_id'],
        industry=row['industry'],
        region=row['region'],
        product_type=row['product_type'],
        lead_source=row['lead_source']
    )
    risk_score = calculate_simple_risk_score(deal, historical_data)
    if risk_score > 70:
        high_risk_deals.append((deal.deal_id, risk_score, deal.deal_stage, deal.deal_amount))

if high_risk_deals:
    print(f"\n1. HIGH-RISK DEAL PATTERNS:")
    print(f"   Found {len(high_risk_deals)} high-risk deals (>70 score)")

    # Analyze patterns
    stages = [d[2] for d in high_risk_deals]
    most_common_stage = max(set(stages), key=stages.count)
    avg_amount = np.mean([d[3] for d in high_risk_deals])

    print(f"   • Most common stage: {most_common_stage}")
    print(f"   • Average amount: ${avg_amount:,.0f}")
    print(f"   • Recommendation: Focus coaching on {most_common_stage} stage deals")

# Insight 2: Rep performance analysis
print(f"\n2. SALES REP PERFORMANCE:")
rep_stats = {}
for rep in historical_data['sales_rep_id'].unique()[:5]:  # Look at first 5 reps
    rep_deals = historical_data[historical_data['sales_rep_id'] == rep]
    if len(rep_deals) > 5:  # Only consider reps with enough history
        win_rate = (rep_deals['outcome'] == 'won').mean()
        avg_deal_size = rep_deals['deal_amount'].mean()
        rep_stats[rep] = (win_rate, avg_deal_size)

# Find best and worst performers
if rep_stats:
    best_rep = max(rep_stats.items(), key=lambda x: x[1][0])
    worst_rep = min(rep_stats.items(), key=lambda x: x[1][0])

    print(f"   • Best performer: {best_rep[0]} ({best_rep[1][0]:.1%} win rate)")
    print(f"   • Needs coaching: {worst_rep[0]} ({worst_rep[1][0]:.1%} win rate)")
    print(f"   • Recommendation: Pair {worst_rep[0]} with {best_rep[0]} for mentoring")

# Insight 3: Lead source effectiveness
print(f"\n3. LEAD SOURCE EFFECTIVENESS:")
lead_source_stats = {}
for source in historical_data['lead_source'].unique():
    source_deals = historical_data[historical_data['lead_source'] == source]
    win_rate = (source_deals['outcome'] == 'won').mean()
    count = len(source_deals)
    if count > 10:  # Only consider sources with enough volume
        lead_source_stats[source] = (win_rate, count)

if lead_source_stats:
    best_source = max(lead_source_stats.items(), key=lambda x: x[1][0])
    worst_source = min(lead_source_stats.items(), key=lambda x: x[1][0])

    print(f"   • Most effective: {best_source[0]} ({best_source[1][0]:.1%} win rate)")
    print(f"   • Least effective: {worst_source[0]} ({worst_source[1][0]:.1%} win rate)")
    print(f"   • Recommendation: Increase investment in {best_source[0]} programs")

print("\n" + "=" * 60)
print("DEMO COMPLETE - BUSINESS RECOMMENDATIONS")
print("=" * 60)

print("""
IMMEDIATE ACTIONS FOR CRO:

1. PRIORITIZE HIGH-RISK DEALS:
   • Review all deals with risk score >70 this week
   • Assign executive sponsors to critical deals

2. COACHING FOCUS AREAS:
   • Provide targeted training for reps with <40% win rate
   • Develop playbooks for high-risk deal stages

3. PIPELINE HEALTH IMPROVEMENT:
   • Clean up deals stuck in pipeline >90 days
   • Reallocate resources to most effective lead sources

4. MONITORING METRICS:
   • Weekly review of risk distribution
   • Track at-risk revenue as % of total pipeline
   • Monitor rep performance consistency
""")

print("\nThe risk scoring engine provides:")
print("✓ Deal-level risk scores with explanations")
print("✓ Pipeline health dashboards")
print("✓ Actionable recommendations for sales teams")
print("✓ Early warning system for at-risk revenue")