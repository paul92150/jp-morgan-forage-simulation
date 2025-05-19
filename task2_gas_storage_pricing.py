import pandas as pd
import numpy as np

def price_storage_contract(
    injection_schedule,       # List of tuples: [(date, volume), ...] for injection (buy)
    withdrawal_schedule,      # List of tuples: [(date, volume), ...] for withdrawal (sell)
    estimate_price_fn,        # Function: estimate_price(date) returns price at that date
    injection_rate,           # Maximum injection volume per day (MMBtu)
    withdrawal_rate,          # Maximum withdrawal volume per day (MMBtu)
    max_storage_volume,       # Maximum volume that can be stored (MMBtu)
    storage_cost_per_month    # Storage cost per unit per month (USD per MMBtu per month)
):
    """
    Computes the net value of a natural gas storage contract.
    
    The function simulates day-by-day cash flows over the contract period.
    - On injection days, you pay the market price (and thus incur a negative cash flow).
    - On withdrawal days, you receive cash at the market price.
    - Each day, you incur storage costs based on the volume held.
    
    Parameters:
        injection_schedule: list of (date, volume) pairs for injections.
        withdrawal_schedule: list of (date, volume) pairs for withdrawals.
        estimate_price_fn: function that returns the market price for a given date.
        injection_rate: maximum volume that can be injected per day.
        withdrawal_rate: maximum volume that can be withdrawn per day.
        max_storage_volume: maximum storage capacity.
        storage_cost_per_month: cost of storage per unit per month.
    
    Returns:
        A dictionary with:
          - 'net_value': Total net cash flow (USD)
          - 'storage_history': A dictionary of {date: storage_volume}
          - 'cash_flow_history': A dictionary of {date: daily cash flow}
    """
    # Convert schedules to DataFrames and ensure dates are datetime objects
    inj_df = pd.DataFrame(injection_schedule, columns=['Date', 'Volume'])
    wdr_df = pd.DataFrame(withdrawal_schedule, columns=['Date', 'Volume'])
    inj_df['Date'] = pd.to_datetime(inj_df['Date'])
    wdr_df['Date'] = pd.to_datetime(wdr_df['Date'])
    
    # Determine the simulation period: from the earliest event to the latest event
    start_date = min(inj_df['Date'].min(), wdr_df['Date'].min())
    end_date = max(inj_df['Date'].max(), wdr_df['Date'].max())
    
    # Create a daily date range covering the entire simulation period
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Group injection and withdrawal volumes by date
    inj_map = inj_df.groupby('Date')['Volume'].sum().to_dict()
    wdr_map = wdr_df.groupby('Date')['Volume'].sum().to_dict()
    
    # Calculate daily storage cost rate (prorated from monthly cost)
    daily_storage_cost_rate = storage_cost_per_month / 30.0
    
    storage_volume = 0.0      # Current storage level (MMBtu)
    total_cash_flow = 0.0     # Total net cash flow (USD)
    
    # Dictionaries to record daily storage and cash flow
    storage_history = {}
    cash_flow_history = {}
    
    # Loop through each day in the simulation period
    for current_date in all_dates:
        # Ensure current_date is a Timestamp
        current_date = pd.to_datetime(current_date)
        
        # Get injection and withdrawal volumes for this day (default 0 if none)
        inj_vol = inj_map.get(current_date, 0)
        wdr_vol = wdr_map.get(current_date, 0)
        
        # Check that daily injection/withdrawal volumes do not exceed limits
        if inj_vol > injection_rate:
            raise ValueError(f"Injection volume {inj_vol} on {current_date.date()} exceeds injection rate {injection_rate}.")
        if wdr_vol > withdrawal_rate:
            raise ValueError(f"Withdrawal volume {wdr_vol} on {current_date.date()} exceeds withdrawal rate {withdrawal_rate}.")
        
        # Process injection (buying gas)
        if inj_vol > 0:
            price = estimate_price_fn(current_date)
            cash_flow_inj = -price * inj_vol  # negative cash flow (cost)
            total_cash_flow += cash_flow_inj
            storage_volume += inj_vol  # add gas to storage
        
        # Process withdrawal (selling gas)
        if wdr_vol > 0:
            if wdr_vol > storage_volume:
                raise ValueError(f"Not enough gas to withdraw on {current_date.date()}: requested {wdr_vol}, available {storage_volume}.")
            price = estimate_price_fn(current_date)
            cash_flow_wdr = price * wdr_vol  # positive cash flow (revenue)
            total_cash_flow += cash_flow_wdr
            storage_volume -= wdr_vol  # remove gas from storage
        
        # Enforce storage capacity constraints
        if storage_volume > max_storage_volume:
            raise ValueError(f"Storage volume {storage_volume} on {current_date.date()} exceeds maximum capacity {max_storage_volume}.")
        if storage_volume < 0:
            raise ValueError(f"Negative storage volume on {current_date.date()}.")
        
        # Apply daily storage cost based on current storage level
        daily_storage_cost = daily_storage_cost_rate * storage_volume
        total_cash_flow -= daily_storage_cost
        
        # Record the storage level and cash flow for this day
        storage_history[current_date] = storage_volume
        cash_flow_history[current_date] = -daily_storage_cost + (-price * inj_vol if inj_vol > 0 else 0) + (price * wdr_vol if wdr_vol > 0 else 0)
    
    return {
        "net_value": total_cash_flow,
        "storage_history": storage_history,
        "cash_flow_history": cash_flow_history
    }

# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    # Define a simple price estimation function (e.g., seasonal variation)
    def estimate_price(date):
        # Base price is $2.5 per MMBtu
        base_price = 2.5
        # Add a seasonal sine-wave component (annual period)
        seasonal_adjustment = 0.5 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        return base_price + seasonal_adjustment

    # Define injection schedule: [(date, volume in MMBtu), ...]
    injection_schedule = [
        ("2025-06-15", 250_000),
        ("2025-07-15", 250_000)
    ]
    
    # Define withdrawal schedule: [(date, volume in MMBtu), ...]
    withdrawal_schedule = [
        ("2026-01-15", 500_000)
    ]
    
    # Set contract parameters
    injection_rate = 250_000          # Maximum injection per day (MMBtu)
    withdrawal_rate = 500_000         # Maximum withdrawal per day (MMBtu)
    max_storage_volume = 500_000       # Maximum storage capacity (MMBtu)
    storage_cost_per_month = 0.5   # Storage cost per MMBtu per month (USD)
    
    # Price the storage contract
    result = price_storage_contract(
        injection_schedule,
        withdrawal_schedule,
        estimate_price,
        injection_rate,
        withdrawal_rate,
        max_storage_volume,
        storage_cost_per_month
    )
    
    print("Contract Pricing Results:")
    print(f"Net Value: ${result['net_value']:,.2f}")

from datetime import datetime

print("June 15, 2025:", estimate_price(datetime(2025, 6, 15)))
print("July 15, 2025:", estimate_price(datetime(2025, 7, 15)))
print("Jan 15, 2026:", estimate_price(datetime(2026, 1, 15)))
