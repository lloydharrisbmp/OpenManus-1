import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def create_example_portfolio_csv():
    """Create a CSV file with example portfolio holdings."""
    print("Creating example portfolio CSV...")
    
    # Define portfolio data
    companies = [
        {"name": "BHP Group Limited", "ticker": "BHP.AX", "sector": "Materials"},
        {"name": "Commonwealth Bank", "ticker": "CBA.AX", "sector": "Financials"},
        {"name": "CSL Limited", "ticker": "CSL.AX", "sector": "Healthcare"},
        {"name": "Fortescue Metals", "ticker": "FMG.AX", "sector": "Materials"},
        {"name": "National Australia Bank", "ticker": "NAB.AX", "sector": "Financials"},
        {"name": "Macquarie Group", "ticker": "MQG.AX", "sector": "Financials"},
        {"name": "Telstra Corporation", "ticker": "TLS.AX", "sector": "Communication Services"},
        {"name": "Woolworths Group", "ticker": "WOW.AX", "sector": "Consumer Staples"},
        {"name": "Wesfarmers Limited", "ticker": "WES.AX", "sector": "Consumer Discretionary"},
        {"name": "Woodside Energy", "ticker": "WDS.AX", "sector": "Energy"},
    ]
    
    # Create portfolio holdings
    portfolio = []
    total_value = 1000000  # $1M portfolio
    
    for company in companies:
        # Assign random weights that sum to approximately 1
        weight = random.uniform(0.05, 0.15)
        value = total_value * weight
        units = round(value / random.uniform(30, 300))
        price = round(value / units, 2)
        
        portfolio.append({
            "company_name": company["name"],
            "ticker": company["ticker"],
            "sector": company["sector"],
            "units": units,
            "price": price,
            "market_value": round(units * price, 2),
            "asset_class": "Equity",
            "purchase_date": (datetime.now() - timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d")
        })
    
    # Add some fixed income assets
    bonds = [
        {"name": "Australian Government Bond", "rate": 3.75, "asset_class": "Fixed Income"},
        {"name": "NAB Fixed Rate Bond", "rate": 4.25, "asset_class": "Fixed Income"},
        {"name": "Queensland Treasury Corp", "rate": 4.15, "asset_class": "Fixed Income"}
    ]
    
    for bond in bonds:
        value = total_value * random.uniform(0.03, 0.08)
        portfolio.append({
            "company_name": bond["name"],
            "ticker": "N/A",
            "sector": "Fixed Income",
            "units": 1,
            "price": round(value, 2),
            "market_value": round(value, 2),
            "asset_class": bond["asset_class"],
            "purchase_date": (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
        })
    
    # Write to CSV file
    filename = os.path.join(UPLOAD_DIR, "example_portfolio.csv")
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["company_name", "ticker", "sector", "units", "price", "market_value", "asset_class", "purchase_date"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in portfolio:
            writer.writerow(row)
    
    print(f"Created: {filename}")
    return filename

def create_example_transactions_csv():
    """Create a CSV file with example transaction history."""
    print("Creating example transaction history CSV...")
    
    # Start date for transactions (6 months ago)
    start_date = datetime.now() - timedelta(days=180)
    
    # Transaction types and their probability weights
    transaction_types = {
        "Dividend": 0.3,
        "Buy": 0.25,
        "Sell": 0.15,
        "Interest": 0.15,
        "Fee": 0.15
    }
    
    # Companies for the transactions
    companies = [
        "BHP.AX", "CBA.AX", "CSL.AX", "FMG.AX", "NAB.AX", 
        "MQG.AX", "TLS.AX", "WOW.AX", "WES.AX", "WDS.AX"
    ]
    
    # Generate transactions
    transactions = []
    for i in range(50):  # Generate 50 transactions
        trans_date = start_date + timedelta(days=random.randint(0, 180))
        trans_type = random.choices(
            list(transaction_types.keys()), 
            weights=list(transaction_types.values())
        )[0]
        
        if trans_type in ["Buy", "Sell", "Dividend"]:
            security = random.choice(companies)
        elif trans_type == "Interest":
            security = "Cash Account"
        else:  # Fee
            security = "Account Fee"
        
        # Generate appropriate amount based on transaction type
        if trans_type == "Buy":
            amount = -round(random.uniform(1000, 10000), 2)
        elif trans_type == "Sell":
            amount = round(random.uniform(1000, 10000), 2)
        elif trans_type == "Dividend":
            amount = round(random.uniform(100, 1000), 2)
        elif trans_type == "Interest":
            amount = round(random.uniform(10, 100), 2)
        else:  # Fee
            amount = -round(random.uniform(10, 50), 2)
        
        transactions.append({
            "transaction_date": trans_date.strftime("%Y-%m-%d"),
            "settlement_date": (trans_date + timedelta(days=2)).strftime("%Y-%m-%d"),
            "transaction_type": trans_type,
            "security": security,
            "amount": amount,
            "balance": 0  # Will calculate running balance later
        })
    
    # Sort by date
    transactions.sort(key=lambda x: datetime.strptime(x["transaction_date"], "%Y-%m-%d"))
    
    # Calculate running balance
    balance = 50000  # Start with $50k
    for trans in transactions:
        balance += trans["amount"]
        trans["balance"] = round(balance, 2)
    
    # Write to CSV file
    filename = os.path.join(UPLOAD_DIR, "example_transactions.csv")
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ["transaction_date", "settlement_date", "transaction_type", "security", "amount", "balance"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in transactions:
            writer.writerow(row)
    
    print(f"Created: {filename}")
    return filename

def create_example_performance_excel():
    """Create an Excel file with portfolio performance data."""
    print("Creating example performance Excel file...")
    
    # Period - monthly data for the past 3 years
    end_date = datetime.now().replace(day=1)
    periods = []
    current_date = end_date
    
    for _ in range(36):  # 3 years
        periods.append(current_date.strftime("%Y-%m"))
        # Move to previous month
        if current_date.month == 1:
            current_date = current_date.replace(year=current_date.year-1, month=12)
        else:
            current_date = current_date.replace(month=current_date.month-1)
    
    # Reverse to get chronological order
    periods.reverse()
    
    # Generate portfolio performance
    portfolio_value = 750000  # Starting value
    portfolio_values = []
    portfolio_returns = []
    benchmark_returns = []
    
    for i, period in enumerate(periods):
        if i > 0:
            # Generate monthly return (between -5% and +5%)
            monthly_return = random.uniform(-0.05, 0.05)
            portfolio_returns.append(round(monthly_return * 100, 2))
            
            # Benchmark return (correlated but different)
            benchmark_monthly = monthly_return + random.uniform(-0.01, 0.01)
            benchmark_returns.append(round(benchmark_monthly * 100, 2))
            
            # Update portfolio value
            portfolio_value = portfolio_value * (1 + monthly_return)
        else:
            # First month, no return yet
            portfolio_returns.append(None)
            benchmark_returns.append(None)
        
        portfolio_values.append(round(portfolio_value, 2))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Period': periods,
        'Portfolio_Value': portfolio_values,
        'Portfolio_Return_Pct': portfolio_returns,
        'Benchmark_Return_Pct': benchmark_returns
    })
    
    # Calculate cumulative returns
    df['Cumulative_Portfolio_Return'] = [None] + [round(((df['Portfolio_Value'][i] / df['Portfolio_Value'][0]) - 1) * 100, 2) for i in range(1, len(df))]
    
    # Fill NaN values
    df = df.fillna("")
    
    # Create asset allocation data
    allocation_data = {
        'Asset_Class': ['Australian Equities', 'International Equities', 'Fixed Income', 'Property', 'Cash'],
        'Allocation_Pct': [35, 30, 20, 10, 5],
        'Target_Pct': [40, 25, 20, 10, 5],
        'Variance_Pct': [-5, 5, 0, 0, 0]
    }
    
    allocation_df = pd.DataFrame(allocation_data)
    
    # Create region allocation data
    region_data = {
        'Region': ['Australia', 'US', 'Europe', 'Asia-Pacific', 'Emerging Markets'],
        'Allocation_Pct': [45, 25, 15, 10, 5]
    }
    
    region_df = pd.DataFrame(region_data)
    
    # Save to Excel with multiple sheets
    filename = os.path.join(UPLOAD_DIR, "portfolio_performance.xlsx")
    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name='Performance', index=False)
        allocation_df.to_excel(writer, sheet_name='Asset Allocation', index=False)
        region_df.to_excel(writer, sheet_name='Regional Exposure', index=False)
    
    print(f"Created: {filename}")
    return filename

if __name__ == "__main__":
    create_example_portfolio_csv()
    create_example_transactions_csv()
    create_example_performance_excel()
    print("Example financial documents created successfully in the 'uploads' directory.")
    print("These can be used for testing the document analyzer functionality.") 