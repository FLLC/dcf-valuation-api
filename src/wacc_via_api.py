# WACC Calculation
import yfinance as yf
import sys 
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
ticker = "AAPL"

def calculates_wacc(ticker: str) -> float:
    """
    in-function functions:
    all take ticker as a arg 
    """
    def get_beta(ticker):
        try:
            data = yf.Ticker(ticker)
            beta = data.info.get('beta', None) 

            if beta is not None:
                return beta
            
            else:
                sys.exit(f"Exiting program due to missing beta for '{ticker}'.")

        except Exception as e:
            sys.exit(f"An error occurred while fetching beta for '{ticker}': {e}")
    def get_equity_value(ticker):
        try:
            data = yf.Ticker(ticker)
            market_cap = data.info.get('marketCap', None)

            if market_cap is not None:
                return market_cap
            
            else:
                sys.exit(f"Exiting program due to missing market cap for '{ticker}'.")

        except Exception as e:
            sys.exit(f"An error occurred while fetching market cap for '{ticker}': {e}")
    def get_total_debt(ticker):
        """Fetches the most recent total debt from the quarterly balance sheet."""
        try:
            stock = yf.Ticker(ticker)
            quarterly_balance_sheet = stock.quarterly_balance_sheet

            # Check for 'Total Debt' in the index
            if 'Total Debt' in quarterly_balance_sheet.index:
                total_debt = quarterly_balance_sheet.loc['Total Debt'].iloc[0]  # Get the most recent total debt
                
                if total_debt is not None:
                    return total_debt
                else:
                    sys.exit(f"Exiting program due to missing total debt for '{ticker}'.")
            else:
                sys.exit(f"Exiting program because 'Total Debt' is not available for '{ticker}'.")

        except Exception as e:
            sys.exit(f"An error occurred while fetching total debt for '{ticker}': {e}")
    def get_effective_tax_rate(ticker):
        """
        Retrieves the effective tax rate directly from the income statement for the given ticker.
        Uses sys.exit to stop the program if the value cannot be fetched.
        """
        try:
            stock = yf.Ticker(ticker)
            income_statement = stock.financials

            # Check if "Tax Rate For Calcs" is in the income statement
            if 'Tax Rate For Calcs' in income_statement.index:
                effective_tax_rate = income_statement.loc['Tax Rate For Calcs'].iloc[0]
                return effective_tax_rate
            else:
                sys.exit(f"Exiting program: 'Tax Rate For Calcs' not found in income statement for '{ticker}'.")

        except Exception as e:
            sys.exit(f"An error occurred while fetching the effective tax rate for '{ticker}': {e}")
    def get_risk_free_rate():
        """ 
        Returns the risk free rate in decimal format. 
        Takes the risk free rate to be the 10 year US treasury note.

        Example: 
        For 3%, it returns 0.03.
        """
        try:
            data = yf.Ticker('^TNX')  # 10-year U.S. Treasury Note
            rate = data.history(period='1d')['Close'].iloc[-1]  # Get the latest closing price
            return rate / 100  # Convert to decimal format
        
        except Exception as e:
            sys.exit(f"An error occurred while fetching the risk-free rate: {e}")
    def get_average_yearly_return(index_ticker='^GSPC'):
        """
        Returns the average yearly return of the selected index or stock.

        The return is in decimal format (e.g., 0.1 for 10%).

        Args:
            index_ticker (str, optional): The ticker symbol of the index or stock. 
                                        Defaults to '^GSPC' for the S&P 500.

        Returns:
            float: The average yearly return as a decimal.
        """
        
        # Calculate the start date as 10 years ago from today
        start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')  # 3650 days is approximately 10 years
        
        # Get today's date
        end_date = datetime.now().strftime('%Y-%m-%d')

        try:
            # Fetch historical data for the specified index or stock
            data = yf.download(index_ticker, start=start_date, end=end_date)

            # Resample to get the yearly data
            yearly_data = data['Adj Close'].resample('YE').last()  # Last adjusted close price of each year

            # Calculate yearly returns
            yearly_returns = yearly_data.pct_change()

            # Calculate average yearly return
            average_yearly_return = yearly_returns.mean()

            return average_yearly_return
        except Exception as e:
            sys.exit(f"An error occurred while fetching data for '{index_ticker}': {e}")
    def calculate_cost_of_debt(ticker, tax_rate):
        """
        Calculates the cost of debt for a given ticker based on interest expense.

        Args:
            ticker (str): The ticker symbol of the company.
            tax_rate (float): The corporate tax rate (in decimal format).

        Returns:
            float: The calculated cost of debt.
        """
        try:
            # Fetch the income statement for the ticker
            stock = yf.Ticker(ticker)
            income_statement = stock.financials

            # Extract interest expense (usually labeled as 'Interest Expense')
            interest_expense = income_statement.loc['Interest Expense'].iloc[0]  # Get the latest interest expense
            if interest_expense is None or interest_expense <= 0 or np.isnan(interest_expense):
                sys.exit(f"Exiting program due to missing or zero interest expense for '{ticker}'.")

            # Get the total debt from the balance sheet
            total_debt = stock.balance_sheet.loc['Total Debt'].iloc[0]  # Get the latest total debt

            if total_debt is None or total_debt <= 0 or np.isnan(interest_expense):
                sys.exit(f"Exiting program due to missing or zero total debt for '{ticker}'.")

            # Calculate the cost of debt using the formula
            cost_of_debt = interest_expense / total_debt * (1 - tax_rate)  # After-tax cost of debt

            return cost_of_debt

        except Exception as e:
            sys.exit(f"An error occurred while calculating cost of debt for '{ticker}': {e}")
    
    # FETCH DATA 
    beta = get_beta(ticker)
    equity_value = get_equity_value(ticker)
    total_debt = get_total_debt(ticker)
    tax_rate = get_effective_tax_rate(ticker)
    r_f = get_risk_free_rate()
    
    # Benchmark for market return (default is SP500)
    avg_yrly_return = get_average_yearly_return()
    # Market risk premium
    market_risk_prem = avg_yrly_return - r_f
    # Cost of Equity
    equity_cost = r_f + beta* market_risk_prem
    # % of debt and of equity
    
    debt_and_equity = equity_value + total_debt
    debt_percentage = total_debt / (debt_and_equity)
    equity_percentage = equity_value / (debt_and_equity)

    # Cost of debt
    debt_cost = calculate_cost_of_debt(ticker, tax_rate)

    return (equity_percentage * equity_cost) + (debt_percentage * debt_cost * (1 - tax_rate))
    

if __name__ == "__main__":
    print(f"{(calculates_wacc(ticker))*100:.2f}%")
    

