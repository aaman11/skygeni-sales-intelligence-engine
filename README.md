
# SkyGeni Sales Intelligence – Decision Insights System

# SkyGeni Risk Scoring Engine

## Overview

SkyGeni Risk Scoring Engine is an AI-powered sales intelligence platform that predicts deal outcomes, identifies at-risk revenue, and provides actionable insights to improve sales performance. The system analyzes historical sales data and current pipeline to generate real-time risk scores and business recommendations.

## Part 1 – Problem Framing

### Real Business Problem
The core business problem is **unpredictable revenue forecasting and preventable deal losses**. Sales teams lack visibility into which deals are at risk and why, leading to:
- Inaccurate quarterly forecasts
- Last-minute surprises in revenue recognition
- Inefficient resource allocation across deals
- Inability to proactively coach on failing deals

### Key Questions for the CRO
1. "Which deals in our pipeline are most likely to be lost, and why?"
2. "How much of next quarter's forecasted revenue is at high risk?"
3. "Which sales reps need immediate coaching intervention?"
4. "What deal patterns historically lead to losses in our business?"
5. "Where should we focus our sales enablement resources this month?"

### Critical Metrics for Win Rate Diagnosis
1. **Deal Velocity**: Time spent in each sales stage
2. **Stage Transition Rates**: Probability of moving from one stage to next
3. **Deal Aging**: Days since last meaningful progress
4. **Rep Performance Variability**: Consistency across similar deals
5. **Product-Industry Fit**: Win rates by industry-product combinations
6. **Competitive Presence**: Win rates when specific competitors are involved

### Key Assumptions
1. Historical data patterns will continue into the future
2. Sales process is consistent enough for meaningful analysis
3. Data quality is sufficient (limited missing critical fields)
4. Deal outcomes are primarily influenced by measurable factors
5. Sales reps will act on provided insights in a timely manner

## Part 4 – Mini System Design: Lightweight Sales Insight & Alert System

### High-Level Architecture
┌─────────────────────────────────────────────────────────┐
│ Data Sources │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ Salesforce│ │ CSV Export│ │ HubSpot │ │
│ └─────┬────┘ └────┬─────┘ └────┬─────┘ │
│ │ │ │ │
└────────┼─────────────┼─────────────┼─────────────────────┘
▼ ▼ ▼
┌─────────────────────────────────────────────────────────┐
│ Data Ingestion Service │
│ ┌─────────────────────────────────────┐ │
│ │ • API Polling (hourly) │ │
│ │ • File Upload Processing │ │
│ │ • Data Validation & Cleansing │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────┐
│ Risk Scoring Engine (Container) │
│ ┌─────────────────────────────────────┐ │
│ │ • Rule-based Scoring Logic │ │
│ │ • Simple ML Model (if data >1000) │ │
│ │ • Real-time Calculation │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────┐
│ Alert Generation Service │
│ ┌─────────────────────────────────────┐ │
│ │ • Threshold-based Alerts │ │
│ │ • Daily Digest Creation │ │
│ │ • Priority Queue Management │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────┘
▼
┌─────────────────────────────────────────────────────────┐
│ Delivery Channels │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ Slack │ │ Email │ │ Webhook │ │
│ │ Bots │ │ Digest │ │ (CRM) │ │
│ └──────────┘ └──────────┘ └──────────┘ │
└─────────────────────────────────────────────────────────┘


### Execution Frequency
- **Data Refresh**: Hourly (during business hours), Nightly (full sync)
- **Risk Scoring**: Real-time on deal creation/update, Batch nightly for all
- **Alert Generation**: Continuous monitoring, triggered by threshold breaches
- **Digest Reports**: Daily (8 AM local time), Weekly (Monday 9 AM)
- **Model Retraining**: Weekly (Sunday night)

### Failure Cases & Limitations

#### **Technical Failure Cases**
1. **CRM API Downtime**: System continues with cached data, alerts on sync failure
2. **Scoring Service Crash**: Fallback to last known scores, degraded insights
3. **Database Corruption**: Daily backups with point-in-time recovery
4. **Alert Storm**: Rate limiting and deduplication to prevent notification fatigue
5. **Model Degradation**: Automatic fallback to rule-based scoring with monitoring alerts

#### **Business Limitations**
1. **Data Quality Dependence**: Garbage in, garbage out - depends on CRM data entry
2. **Novel Situations**: Cannot predict unprecedented market shifts or black swan events
3. **Human Factor**: Cannot account for exceptional salesperson relationships
4. **Signal Lag**: Latest competitive intel may not be in CRM yet
5. **Confidential Deals**: Some deal details may be kept offline for strategic reasons

#### **Scalability Constraints**
- **Current**: ~10,000 deals, 100 users, <1s response time
- **Next Tier**: ~100,000 deals requires sharding and queue optimization
- **Enterprise**: >1M deals needs distributed processing and ML pipeline redesign

## Part 5 – Reflection

### Weakest Assumptions in My Solution
1. **Data Completeness**: Assuming CRM data is consistently updated and contains all relevant fields. In reality, sales reps often keep critical information in emails or notes.
2. **Causality vs Correlation**: Assuming patterns from historical data imply causation. A deal stalling at negotiation might be strategic, not problematic.
3. **Uniform Sales Process**: Assuming all reps follow the same process and stage definitions. Reality has significant individual variation.
4. **Immediate Actionability**: Assuming insights will be acted upon promptly. Organizational inertia and competing priorities often delay action.

### What Would Break in Real-World Production
1. **Data Schema Changes**: CRM fields get renamed or repurposed, breaking data ingestion.
2. **Edge Cases**: Special deal types (partnerships, acquisitions) that don't fit standard patterns.
3. **Feedback Loops**: Sales reps gaming the system once they understand scoring factors.
4. **Performance**: Real-time scoring for thousands of concurrent users during month-end.
5. **Explainability Demands**: Stakeholders demanding explanations for every "high-risk" classification.

### What Would I Build Next (1 Month Timeline)
**Phase 1: "Deal Health Dashboard" (Week 1-2)**
- Interactive dashboard with drill-down capabilities
- Team-level and individual rep views
- Simple what-if analysis tools

**Phase 2: "Automated Coaching Assistant" (Week 3)**
- Personalized action plans for each high-risk deal
- Template emails, call scripts, battle cards
- Integration with sales engagement platforms

**Phase 3: "Predictive Resource Allocation" (Week 4)**
- Suggest which deals need executive involvement
- Recommend best rep for specific deal types
- Forecast support resource needs

### Least Confident Part of My Solution
**The ensemble model combining rule-based and ML scoring** - While theoretically sound, getting the weighting right between:
- Interpretable business rules (transparent but rigid)
- ML patterns (flexible but opaque)

In practice, this requires continuous calibration and risks either:
1. Overweighting rules (missing subtle patterns)
2. Overweighting ML (losing business intuition)

The feedback mechanism for tuning this balance would need careful design and likely several iterations to get right.

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/aaman11/skygeni-sales-intelligence-engine.git
cd skygeni-risk-engine

# Install dependencies
pip install -r requirements.txt

# Generate sample data and run demo
python decision_engine.py

# Run EDA analysis
python eda.py
