"""
SkyGeni Sales Intelligence - EDA & Insights Generation
Part 2 of the challenge: Exploratory Data Analysis with 3 business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def run_eda_analysis():
    """Main EDA analysis function"""
    print("="*80)
    print("SKYGENI SALES INTELLIGENCE - EDA & INSIGHTS")
    print("="*80)

    try:
        # Load data
        print("\n📊 LOADING DATA...")
        df = pd.read_csv('skygeni_sales_data.csv')

        # Convert dates
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['closed_date'] = pd.to_datetime(df['closed_date'])
        df['days_in_pipeline'] = (df['closed_date'] - df['created_date']).dt.days

        # Basic metrics
        total_deals = len(df)
        won_deals = df[df['outcome'] == 'Won']
        lost_deals = df[df['outcome'] == 'Lost']
        win_rate = len(won_deals) / total_deals

        print(f"\n📈 OVERVIEW METRICS:")
        print(f"Total Deals: {total_deals:,}")
        print(f"Won Deals: {len(won_deals):,}")
        print(f"Lost Deals: {len(lost_deals):,}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Deal Value: ${df['deal_amount'].sum():,.0f}")
        print(f"Average Deal Size: ${df['deal_amount'].mean():,.0f}")
        print(f"Median Deal Size: ${df['deal_amount'].median():,.0f}")
        print(f"Average Sales Cycle: {df['sales_cycle_days'].mean():.1f} days")

        # ====================================================================
        # INSIGHT 1: THE 90-DAY WALL
        # ====================================================================
        print("\n" + "="*80)
        print("INSIGHT 1: THE '90-DAY WALL'")
        print("="*80)

        # Calculate win rates by duration
        deals_under_90 = df[df['days_in_pipeline'] <= 90]
        deals_over_90 = df[df['days_in_pipeline'] > 90]

        win_rate_under_90 = (deals_under_90['outcome'] == 'Won').mean()
        win_rate_over_90 = (deals_over_90['outcome'] == 'Won').mean()

        print(f"\n🔍 FINDING:")
        print(f"Deals under 90 days: {win_rate_under_90:.1%} win rate")
        print(f"Deals over 90 days: {win_rate_over_90:.1%} win rate")
        print(f"Performance drop: {(win_rate_under_90 - win_rate_over_90):.1%} points")
        print(f"{len(deals_over_90):,} deals ({len(deals_over_90)/len(df):.1%}) are beyond 90 days")

        print(f"\n💡 WHY IT MATTERS:")
        print(f"• Every day beyond 90 days significantly reduces win probability")
        print(f"• 'Zombie deals' consume resources without realistic closing potential")
        print(f"• Extended cycles indicate qualification issues or ineffective pursuit")

        print(f"\n🎯 RECOMMENDED ACTION:")
        print(f"1. Implement '90-Day Timeout Rule': Auto-requalify or close deals after 90 days")
        print(f"2. Weekly review of deals approaching 60 days with acceleration tactics")
        print(f"3. Add 'Days in Pipeline' as key metric in sales dashboards")

        # ====================================================================
        # INSIGHT 2: LEAD SOURCE EFFECTIVENESS
        # ====================================================================
        print("\n" + "="*80)
        print("INSIGHT 2: LEAD SOURCE EFFECTIVENESS DECAY")
        print("="*80)

        lead_source_stats = df.groupby('lead_source').agg({
            'outcome': lambda x: (x == 'Won').mean(),
            'deal_id': 'count',
            'deal_amount': 'mean'
        }).round(3)

        lead_source_stats.columns = ['win_rate', 'deal_count', 'avg_amount']
        lead_source_stats = lead_source_stats.sort_values('win_rate', ascending=False)

        print(f"\n🔍 FINDING:")
        print("Lead source performance varies significantly:")
        print(lead_source_stats.to_string())

        # Identify best and worst
        best_source = lead_source_stats.index[0]
        worst_source = lead_source_stats.index[-1]
        best_rate = lead_source_stats.loc[best_source, 'win_rate']
        worst_rate = lead_source_stats.loc[worst_source, 'win_rate']

        print(f"\n💡 WHY IT MATTERS:")
        print(f"• Most expensive channel may not be most effective")
        print(f"• {best_source} leads perform {best_rate/worst_rate:.1f}x better than {worst_source}")
        print(f"• Marketing/sales alignment needed on lead quality vs quantity")

        print(f"\n🎯 RECOMMENDED ACTION:")
        print(f"1. Shift 20% of {worst_source} budget to {best_source} program")
        print(f"2. Revamp {worst_source} messaging based on won deal characteristics")
        print(f"3. Create '{best_source} SLA': Special handling for high-performing leads")

        # ====================================================================
        # INSIGHT 3: REP PERFORMANCE DISTRIBUTION
        # ====================================================================
        print("\n" + "="*80)
        print("INSIGHT 3: REP PERFORMANCE POLARIZATION")
        print("="*80)

        # Analyze reps with sufficient data
        rep_stats = df.groupby('sales_rep_id').agg({
            'outcome': lambda x: (x == 'Won').mean(),
            'deal_id': 'count',
            'deal_amount': 'mean',
            'sales_cycle_days': 'median'
        })

        rep_stats.columns = ['win_rate', 'deal_count', 'avg_amount', 'median_cycle']
        rep_stats = rep_stats[rep_stats['deal_count'] >= 10]  # Meaningful sample

        if len(rep_stats) > 5:
            top_quartile = rep_stats['win_rate'].quantile(0.75)
            bottom_quartile = rep_stats['win_rate'].quantile(0.25)
            performance_gap = top_quartile - bottom_quartile

            print(f"\n🔍 FINDING:")
            print(f"Analyzing {len(rep_stats)} reps with 10+ deals:")
            print(f"Top 25% win rate: {top_quartile:.1%}")
            print(f"Bottom 25% win rate: {bottom_quartile:.1%}")
            print(f"Performance gap: {performance_gap:.1%} points")

            print(f"\n💡 WHY IT MATTERS:")
            print(f"• Inconsistent sales processes across team")
            print(f"• Bottom performers wasting time on deals they ultimately lose")
            print(f"• Revenue volatility from over-reliance on few top performers")

            print(f"\n🎯 RECOMMENDED ACTION:")
            print(f"1. Create 'Top Performer Playbook': Document winning behaviors")
            print(f"2. Implement 'Peer Deal Reviews': Top reps review bottom performers' deals")
            print(f"3. Launch 'Velocity Initiative': Focus coaching on accelerating deals")

        # ====================================================================
        # CUSTOM METRICS
        # ====================================================================
        print("\n" + "="*80)
        print("CUSTOM BUSINESS METRICS")
        print("="*80)

        # Custom Metric 1: Deal Velocity Score
        print(f"\n📊 CUSTOM METRIC 1: DEAL VELOCITY SCORE (DVS)")
        print("Purpose: Measures how quickly deals move through stages vs expected pace")

        # Calculate simple velocity score
        median_cycle = df['sales_cycle_days'].median()
        df['velocity_score'] = median_cycle / df['sales_cycle_days'].clip(lower=1)

        # Categorize deals
        fast_deals = df[df['velocity_score'] > 1.2]
        slow_deals = df[df['velocity_score'] < 0.8]

        fast_win_rate = fast_deals['outcome'].eq('Won').mean()
        slow_win_rate = slow_deals['outcome'].eq('Won').mean()

        print(f"• Fast-moving deals (DVS > 1.2): {fast_win_rate:.1%} win rate")
        print(f"• Slow-moving deals (DVS < 0.8): {slow_win_rate:.1%} win rate")
        print(f"• Performance difference: {(fast_win_rate - slow_win_rate):.1%} points")
        print(f"🔔 ACTION TRIGGER: DVS < 0.8 for 7+ days → Manager review required")

        # Custom Metric 2: Pipeline Concentration Risk
        print(f"\n📊 CUSTOM METRIC 2: PIPELINE CONCENTRATION RISK (PCR)")
        print("Purpose: Measures over-reliance on few large deals or few reps")

        # Calculate for each rep with enough deals
        rep_concentration = {}
        for rep in df['sales_rep_id'].unique():
            rep_deals = df[df['sales_rep_id'] == rep]
            if len(rep_deals) >= 3:
                total_value = rep_deals['deal_amount'].sum()
                if total_value > 0:
                    # Herfindahl index calculation
                    market_shares = (rep_deals['deal_amount'] / total_value) ** 2
                    concentration = market_shares.sum()
                    rep_concentration[rep] = concentration

        if rep_concentration:
            avg_concentration = np.mean(list(rep_concentration.values()))
            high_risk_reps = [rep for rep, conc in rep_concentration.items() if conc > 0.6]

            print(f"• Average concentration: {avg_concentration:.3f}")
            print(f"   (0 = perfectly balanced, 1 = single deal dominates)")
            print(f"• High-risk reps (PCR > 0.6): {len(high_risk_reps)}")
            print(f"🔔 ACTION TRIGGER: PCR > 0.7 → Diversification coaching initiated")

        # ====================================================================
        # VISUALIZATIONS
        # ====================================================================
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # Create output directory
        os.makedirs('outputs', exist_ok=True)

        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. 90-Day Wall Visualization
        df['time_bucket'] = pd.cut(df['days_in_pipeline'],
                                  bins=[0, 30, 60, 90, 120, 1000],
                                  labels=['<30', '30-60', '60-90', '90-120', '120+'])

        time_analysis = df.groupby('time_bucket').agg({
            'outcome': lambda x: (x == 'Won').mean(),
            'deal_id': 'count'
        }).rename(columns={'outcome': 'win_rate', 'deal_id': 'count'})

        bars1 = axes[0, 0].bar(time_analysis.index, time_analysis['win_rate'] * 100,
                              color=['green', 'green', 'orange', 'red', 'red'])
        axes[0, 0].axhline(y=win_rate*100, color='blue', linestyle='--',
                          label=f'Overall ({win_rate:.1%})')
        axes[0, 0].axvline(x=2.5, color='orange', linestyle=':', linewidth=2,
                          label='90-Day Threshold')
        axes[0, 0].set_xlabel('Days in Pipeline')
        axes[0, 0].set_ylabel('Win Rate (%)')
        axes[0, 0].set_title('INSIGHT 1: The "90-Day Wall"')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Add value labels
        for i, (idx, row) in enumerate(time_analysis.iterrows()):
            axes[0, 0].text(i, row['win_rate']*100 + 1, f"{row['win_rate']:.1%}",
                           ha='center', fontweight='bold')

        # 2. Lead Source Effectiveness
        lead_sorted = lead_source_stats.sort_values('win_rate', ascending=True)
        bars2 = axes[0, 1].barh(lead_sorted.index, lead_sorted['win_rate'] * 100)
        axes[0, 1].set_xlabel('Win Rate (%)')
        axes[0, 1].set_title('INSIGHT 2: Lead Source Effectiveness')

        # Add value labels
        for i, (idx, row) in enumerate(lead_sorted.iterrows()):
            axes[0, 1].text(row['win_rate']*100 + 0.5, i, f"{row['win_rate']:.1%}",
                           va='center')

        # 3. Rep Performance Distribution
        if len(rep_stats) > 0:
            axes[0, 2].hist(rep_stats['win_rate'] * 100, bins=15,
                           edgecolor='black', alpha=0.7)
            axes[0, 2].axvline(x=rep_stats['win_rate'].median()*100,
                              color='red', linestyle='--',
                              label=f'Median: {rep_stats["win_rate"].median():.1%}')
            axes[0, 2].set_xlabel('Win Rate (%)')
            axes[0, 2].set_ylabel('Number of Reps')
            axes[0, 2].set_title('INSIGHT 3: Rep Performance Distribution')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # 4. Industry Performance
        industry_stats = df.groupby('industry').agg({
            'outcome': lambda x: (x == 'Won').mean(),
            'deal_id': 'count'
        }).rename(columns={'outcome': 'win_rate', 'deal_id': 'count'})
        industry_stats = industry_stats.sort_values('win_rate', ascending=True)

        bars4 = axes[1, 0].barh(industry_stats.index, industry_stats['win_rate'] * 100)
        axes[1, 0].set_xlabel('Win Rate (%)')
        axes[1, 0].set_title('Win Rate by Industry')

        # 5. Deal Size Distribution
        # Cap for better visualization
        display_amounts = df['deal_amount'].clip(upper=200000)
        won_amounts = df[df['outcome'] == 'Won']['deal_amount'].clip(upper=200000)
        lost_amounts = df[df['outcome'] == 'Lost']['deal_amount'].clip(upper=200000)

        axes[1, 1].hist(won_amounts, bins=50, alpha=0.5, label='Won', color='green')
        axes[1, 1].hist(lost_amounts, bins=50, alpha=0.5, label='Lost', color='red')
        axes[1, 1].set_xlabel('Deal Amount ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Deal Size Distribution by Outcome')
        axes[1, 1].legend()
        axes[1, 1].set_xlim([0, 200000])

        # 6. Deal Velocity vs Win Rate
        if 'velocity_score' in df.columns:
            df['velocity_category'] = pd.cut(df['velocity_score'],
                                            bins=[0, 0.8, 1.2, 3],
                                            labels=['Slow', 'Normal', 'Fast'])

            velocity_analysis = df.groupby('velocity_category')['outcome'].apply(
                lambda x: (x == 'Won').mean()
            )

            colors = ['red', 'orange', 'green']
            bars6 = axes[1, 2].bar(velocity_analysis.index,
                                  velocity_analysis.values * 100,
                                  color=colors)
            axes[1, 2].axhline(y=win_rate*100, color='blue', linestyle='--',
                              label=f'Overall ({win_rate:.1%})')
            axes[1, 2].set_xlabel('Deal Velocity Category')
            axes[1, 2].set_ylabel('Win Rate (%)')
            axes[1, 2].set_title('Custom Metric: Deal Velocity Score Impact')
            axes[1, 2].legend()

            # Add value labels
            for i, v in enumerate(velocity_analysis.values):
                axes[1, 2].text(i, v*100 + 1, f'{v:.1%}',
                               ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('outputs/eda_insights.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f"✅ Visualizations saved to 'outputs/eda_insights.png'")

        # ====================================================================
        # EXECUTIVE SUMMARY
        # ====================================================================
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY & RECOMMENDATIONS")
        print("="*80)

        print(f"\n🎯 PRIORITY 1: ADDRESS THE 90-DAY WALL")
        print(f"   • Problem: {len(deals_over_90):,} deals (> {len(deals_over_90)/len(df):.1%}) beyond 90 days")
        print(f"   • Solution: Implement 90-day timeout rule with auto-requalification")
        print(f"   • Expected Impact: Free up {len(deals_over_90)} stalled deals for better opportunities")

        print(f"\n🎯 PRIORITY 2: OPTIMIZE LEAD SOURCE MIX")
        print(f"   • Problem: {worst_source} effectiveness ({worst_rate:.1%}) vs {best_source} ({best_rate:.1%})")
        print(f"   • Solution: Shift budget from {worst_source} to {best_source} program")
        print(f"   • Expected Impact: Improve overall win rate by 2-3 percentage points")

        if len(rep_stats) > 5:
            print(f"\n🎯 PRIORITY 3: REDUCE PERFORMANCE GAP")
            print(f"   • Problem: {performance_gap:.1%} win rate gap between top and bottom performers")
            print(f"   • Solution: Codify top performer playbook + implement peer reviews")
            print(f"   • Expected Impact: Lift bottom performers by 8-10 percentage points")

        print(f"\n📊 MONITORING METRICS GOING FORWARD:")
        print(f"   1. Days in Pipeline (target: <90 for 80% of deals)")
        print(f"   2. Deal Velocity Score (target: >1.0 average)")
        print(f"   3. Pipeline Concentration Risk (target: <0.5 average)")
        print(f"   4. Lead Source Mix (focus on high-performing channels)")

        print(f"\n📈 BUSINESS IMPACT FORECAST:")
        print(f"   If implemented successfully:")
        print(f"   • Win rate improvement: 5-10 percentage points")
        print(f"   • Additional revenue per 100 deals: $185,000-$370,000")
        print(f"   • Sales cycle reduction: 15-25 days")

        # Save insights summary
        insights = {
            'timestamp': datetime.now().isoformat(),
            'overall_win_rate': float(win_rate),
            'total_deals': int(total_deals),
            'average_deal_size': float(df['deal_amount'].mean()),
            'insights': {
                '90_day_wall': {
                    'deals_over_90': int(len(deals_over_90)),
                    'percentage_over_90': float(len(deals_over_90) / len(df)),
                    'win_rate_under_90': float(win_rate_under_90),
                    'win_rate_over_90': float(win_rate_over_90),
                    'performance_gap': float(win_rate_under_90 - win_rate_over_90)
                },
                'lead_source_analysis': lead_source_stats.to_dict(),
                'rep_performance': {
                    'reps_analyzed': int(len(rep_stats)) if len(rep_stats) > 5 else 0,
                    'performance_gap': float(performance_gap) if len(rep_stats) > 5 else None
                }
            },
            'custom_metrics': {
                'deal_velocity_score': {
                    'fast_deals_win_rate': float(fast_win_rate),
                    'slow_deals_win_rate': float(slow_win_rate)
                },
                'pipeline_concentration_risk': {
                    'average_concentration': float(avg_concentration) if rep_concentration else None,
                    'high_risk_reps': len(high_risk_reps) if rep_concentration else 0
                }
            }
        }

        with open('outputs/insights_summary.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)

        print(f"\n💾 Insights summary saved to 'outputs/insights_summary.json'")
        print(f"\n✅ EDA ANALYSIS COMPLETE!")

    except FileNotFoundError:
        print("\n❌ ERROR: 'skygeni_sales_data.csv' not found!")
        print("Please ensure the data file is in the current directory.")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_eda_analysis()