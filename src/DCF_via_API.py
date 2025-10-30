# DCF Main Body
import requests
import json
import io
import pandas as pd
import sys
from wacc_via_API import calculates_wacc
import yfinance as yf
import logging
import numpy as np


def DCF_pricing_calculator(
    ticker: str,
    revenue_growth: float = 0.1,
    wacc: float = None,
    n_periods: float = 6,
    tgr: float = 0.02,
) -> tuple:

    # Billions (10^9) / Millions (10^6)
    abs_units = 1_000_000_000
    """
    in_function functions:
    """

    def get_shares_outstanding(ticker):
        # Fetch stock data
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get market cap and current share price
        market_cap = info.get("marketCap", None)  # Market capitalization
        current_price = info.get("currentPrice", None)  # Current stock price

        if market_cap and current_price:
            shares_outstanding = market_cap / current_price
            return (shares_outstanding, current_price)
        else:
            raise ValueError(
                "Unable to fetch market cap or current price from Yahoo Finance."
            )

    def make_request(isin, data):
        """
        Makes an API request to retrieve financial statements for a given ISIN.
        Returns balance sheet, cash flow, and income statement DataFrames, or None if the request fails.
        """
        base_url = "https://iveurope.eu/git/api.php?"
        base_url += "ISIN=" + isin
        base_url += "&data=" + data

        try:
            # Make the request
            response = requests.get(base_url)
            response.raise_for_status()  # Raise an error if the request failed

            # Parse JSON
            json_response = response.json()

            # Ensure required data exists in the response
            if "Financials" in json_response and all(
                key in json_response["Financials"]
                for key in ["Balance_sheet", "Cash_flow", "Income_statement"]
            ):

                # Financial statements in csv format
                balance_sheet_csv = json_response["Financials"]["Balance_sheet"]
                cash_flow_csv = json_response["Financials"]["Cash_flow"]
                income_statement_csv = json_response["Financials"]["Income_statement"]

                # Convert CSV to DataFrames
                balance_sheet = pd.read_csv(io.StringIO(balance_sheet_csv))
                cash_flow = pd.read_csv(io.StringIO(cash_flow_csv))
                income_statement = pd.read_csv(io.StringIO(income_statement_csv))

                return balance_sheet, cash_flow, income_statement

            else:
                logging.error(f"Request failed: {e}")
                return None, None, None

        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None, None, None

    def fill_moving_average(row_name, periods=4):
        """
        Fills the future columns of the specified row with a moving average of the given number of periods.

        :param row_name: Name of the row to process.
        :param periods: Number of periods for the moving average (default is 4).
        """
        # Identify year columns (assume columns representing years are 4-digit strings)
        year_columns = [
            col
            for col in financial_projections_df.columns
            if col.isdigit() and len(col) == 4
        ]
        last_year_col = max(year_columns, key=int)  # Get the most recent year column

        # Iterate over the future columns after the most recent year column
        for i in range(
            financial_projections_df.columns.get_loc(last_year_col) + 1,
            len(financial_projections_df.columns),
        ):  # Start after the last year column
            current_col = financial_projections_df.columns[i]
            # Fetch the previous 'periods' values from the columns
            past_values = financial_projections_df.loc[
                row_name, financial_projections_df.columns[i - periods : i]
            ]
            # Calculate the moving average
            moving_avg = past_values.mean()
            # Round the moving average to 2 decimal places and fill the future columns
            financial_projections_df.loc[row_name, current_col] = round(moving_avg, 2)

    if ticker == "EXIT":
        sys.exit("EXIT was typed")

    # Load the pre-cleaned JSON data
    try:
        with open("Cleaned_ISIN_tickers.json") as file:
            data = json.load(file)
    except FileNotFoundError:
        sys.exit("Error: JSON file not found.")
    except json.JSONDecodeError:
        sys.exit("Error: Failed to decode JSON file.")

    # Check if the ticker exists in the JSON file
    if ticker in data:
        isin = data[ticker]["isin"]
        ticker = ticker.upper()
    else:
        sys.exit("Ticker not found in the JSON file")

    # In case of no user input the wacc is fetched fro calculates_wacc
    if wacc is None:
        wacc1 = calculates_wacc(ticker)
        if np.isnan(wacc1):
            sys.exit("WACC cannot be calculated")
        else:
            wacc = wacc1
    # Attempt to retrieve data from the API
    balance_sheet, cash_flow, income_statement = make_request(isin, data="financials")

    # Check if data was successfully retrieved
    if balance_sheet is None or cash_flow is None or income_statement is None:
        sys.exit("Failed to retrieve data from the API")

    ### BS: Balance sheet ###
    ## Access the value for "Total Debt"
    # Checking for "Total Debt" in the 'Fiscal Period' column (with space trimming and case insensitivity)
    if any(balance_sheet["Fiscal Period"].str.strip().str.lower() == "total debt"):

        try:
            total_debt = balance_sheet.loc[
                balance_sheet["Fiscal Period"].str.strip().str.lower() == "total debt",
                balance_sheet.columns[1],
            ].values[0]
        except IndexError:
            sys.exit(
                '"Total Debt" row in Balance Sheet found, but value could not be retrieved.'
            )
        except KeyError:
            sys.exit('"Total Debt" column was not found in the Balance Sheet.')
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")

    else:
        # USE AN ALTERNATIVE VALUE FOR THE DEBT?
        logging.warning('"Total Debt" was not found in the Balance Sheet')

    ## Access the value for "Cash"
    # Checking for "Cash" or "Cash & Equivalents" with space trimming and case insensitivity
    if any(balance_sheet["Fiscal Period"].str.strip().str.lower() == "cash") or any(
        balance_sheet["Fiscal Period"].str.strip().str.lower() == "cash & equivalents"
    ):

        # Attempt to find either "Cash" or "Cash & Equivalents"
        if any(balance_sheet["Fiscal Period"].str.strip().str.lower() == "cash"):
            try:
                cash_last_period = balance_sheet.loc[
                    balance_sheet["Fiscal Period"].str.strip().str.lower() == "cash",
                    balance_sheet.columns[1],
                ].values[0]
            except IndexError:
                sys.exit(
                    '"Cash" row in Balance Sheet found, but value could not be retrieved.'
                )
            except KeyError:
                sys.exit('"Cash" column was not found in the Balance Sheet.')
            except Exception as e:
                sys.exit(f"An unexpected error occurred: {e}")

        elif any(
            balance_sheet["Fiscal Period"].str.strip().str.lower()
            == "cash & equivalents"
        ):
            print('"Cash & Equivalents" will be used instead of "Cash"')
            try:
                cash_last_period = balance_sheet.loc[
                    balance_sheet["Fiscal Period"].str.strip().str.lower()
                    == "cash & equivalents",
                    balance_sheet.columns[1],
                ].values[0]
            except IndexError:
                sys.exit(
                    '"Cash & Equivalents" row in Balance Sheet found, but value could not be retrieved.'
                )
            except KeyError:
                sys.exit(
                    '"Cash & Equivalents" column was not found in the Balance Sheet.'
                )
            except Exception as e:
                sys.exit(f"An unexpected error occurred: {e}")

    else:
        print('"Cash" or "Cash & Equivalents" was not found in the Balance Sheet')

    ### CF: Cash Flow ###
    ## Access the value for "Depreciation/Depletion"
    # Checking for "Depreciation/​Depletion" in the 'Fiscal Period' column (with space trimming and case insensitivity)
    if any(
        cash_flow["Fiscal Period"].str.strip().str.lower() == "depreciation/​depletion"
    ):

        try:
            # Retrieve the depreciation values for the row "Depreciation/​Depletion"
            depreciation_per_year = (
                cash_flow.loc[
                    cash_flow["Fiscal Period"].str.strip().str.lower()
                    == "depreciation/​depletion"
                ]
                .iloc[0, 1:]
                .values
            )

            # Convert values to numeric, with invalid values coerced to NaN
            depreciation_per_year = pd.to_numeric(
                depreciation_per_year, errors="coerce"
            )

            # Reverse the years
            depreciation_per_year = depreciation_per_year[::-1]  # Reverse the years

        except IndexError:
            sys.exit(
                '"Depreciation/Depletion" row in Cash Flow found, but value could not be retrieved.'
            )
        except KeyError:
            sys.exit('"Depreciation/​Depletion" column was not found in the Cash Flow.')
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")

    else:
        # Alternative for missing "Depreciation/ ​Depletion"
        print('"Depreciation/​Depletion" was not found in the Cash Flow')

    ## Access the value for "Capital Expenditures"
    # Checking for "Capital Expenditures" in the 'Fiscal Period' column (with space trimming and case insensitivity)
    if any(
        cash_flow["Fiscal Period"].str.strip().str.lower() == "capital expenditures"
    ):

        try:
            # Retrieve the capital expenditures values for the row "Capital Expenditures"
            capex_per_year = (
                cash_flow.loc[
                    cash_flow["Fiscal Period"].str.strip().str.lower()
                    == "capital expenditures"
                ]
                .iloc[0, 1:]
                .values
            )

            # Convert values to numeric, with invalid values coerced to NaN
            capex_per_year = pd.to_numeric(capex_per_year, errors="coerce")

            # Reverse the years
            capex_per_year = capex_per_year[::-1]  # Reverse the years

        except IndexError:
            sys.exit(
                '"Capital Expenditures" row in Cash Flow found, but value could not be retrieved.'
            )
        except KeyError:
            sys.exit('"Capital Expenditures" column was not found in the Cash Flow.')
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")

    else:
        # Alternative for missing "Capital Expenditures"
        print('"Capital Expenditures" was not found in the Cash Flow')

    ## Access the value for "Changes in Working Capital"
    # Checking for "Changes in Working Capital" in the 'Fiscal Period' column (with space trimming and case insensitivity)
    if any(
        cash_flow["Fiscal Period"].str.strip().str.lower()
        == "changes in working capital"
    ):

        try:
            # Retrieve the changes in working capital values for the row "Changes in Working Capital"
            nwc_per_year = (
                cash_flow.loc[
                    cash_flow["Fiscal Period"].str.strip().str.lower()
                    == "changes in working capital"
                ]
                .iloc[0, 1:]
                .values
            )

            # Convert values to numeric, with invalid values coerced to NaN
            nwc_per_year = pd.to_numeric(nwc_per_year, errors="coerce")

            # Reverse the years
            nwc_per_year = nwc_per_year[::-1]  # Reverse the years

        except IndexError:
            sys.exit(
                '"Changes in Working Capital" row in Cash Flow found, but value could not be retrieved.'
            )
        except KeyError:
            sys.exit(
                '"Changes in Working Capital" column was not found in the Cash Flow.'
            )
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")

    else:
        # Alternative for missing "Changes in Working Capital"
        print('"Changes in Working Capital" was not found in the Cash Flow')

    ### IS: Income Statement ###
    ## Access the value for "Total Revenue"
    # Checking for "Total Revenue" in the 'Fiscal Period' column (with space trimming and case insensitivity)
    if any(
        income_statement["Fiscal Period"].str.strip().str.lower() == "total revenue"
    ):
        try:
            # Retrieve the total revenue values for the row "Total Revenue"
            total_revenue = (
                income_statement.loc[
                    income_statement["Fiscal Period"].str.strip().str.lower()
                    == "total revenue"
                ]
                .iloc[0, 1:]
                .values
            )

            # Convert values to numeric, with invalid values coerced to NaN
            total_revenue = pd.to_numeric(total_revenue, errors="coerce")

            # Reverse the years
            total_revenue = total_revenue[::-1]  # Reverse the years

        except IndexError:
            sys.exit(
                '"Total Revenue" row in Income Statement found, but value could not be retrieved.'
            )
        except KeyError:
            sys.exit('"Total Revenue" column was not found in the Income Statement.')
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")

    else:
        # Alternative for missing "Total Revenue"
        print('"Total Revenue" was not found in the Income Statement')

    ## Access the value for "Provision for Income Taxes"
    # Checking for "Provision for Income Taxes" in the 'Fiscal Period' column (with space trimming and case insensitivity)
    if any(
        income_statement["Fiscal Period"].str.strip().str.lower()
        == "provision for income taxes"
    ):
        try:
            tax_provision = (
                income_statement.loc[
                    income_statement["Fiscal Period"].str.strip().str.lower()
                    == "provision for income taxes"
                ]
                .iloc[0, 1:]
                .values
            )
            tax_provision = pd.to_numeric(tax_provision, errors="coerce")
            tax_provision = tax_provision[::-1]  # Reverse the years

        except IndexError:
            sys.exit(
                '"Provision for Income Taxes" row in Income Statement found, but value could not be retrieved.'
            )
        except KeyError:
            sys.exit(
                '"Provision for Income Taxes" column was not found in the Income Statement.'
            )
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")

    else:
        # Alternative for missing "Provision for Income Taxes"
        print('"Provision for Income Taxes" was not found in the Income Statement')

    ## Access the value for "Operating Income"
    # Checking for "Operating Income" in the 'Fiscal Period' column (with space trimming and case insensitivity)
    if any(
        income_statement["Fiscal Period"].str.strip().str.lower() == "operating income"
    ):
        try:
            ebit = (
                income_statement.loc[
                    income_statement["Fiscal Period"].str.strip().str.lower()
                    == "operating income"
                ]
                .iloc[0, 1:]
                .values
            )
            ebit = pd.to_numeric(ebit, errors="coerce")
            ebit = ebit[::-1]  # Reverse the years
        except IndexError:
            sys.exit(
                '"Operating Income" row in Income Statement found, but value could not be retrieved.'
            )
        except KeyError:
            sys.exit('"Operating Income" column was not found in the Income Statement.')
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")
    else:
        # Alternative for missing "Operating Income"
        print('"Operating Income" was not found in the Income Statement')

    ### df creation ###
    ## Access the column names: Years
    # Checking if 'years' row exists and can be reversed
    if len(income_statement.columns) > 1:
        try:
            # Reverse the 'years' column (already done earlier)
            years = income_statement.columns[1:]
            years = years[::-1]  # Reverse the years
            years = years.tolist()
        except IndexError:
            sys.exit("IndexError: Could not reverse the 'years' column as expected.")
        except Exception as e:
            sys.exit(f"An unexpected error occurred: {e}")
    else:
        # Alternative?
        sys.exit("No 'years' column found in the Income Statement")

    # Create the dict for the data
    data = {
        "Total Revenue": total_revenue,
        "Operating Income (EBIT)": ebit,
        "Provision for Income Taxes": tax_provision,
        "Depreciation/Depletion": depreciation_per_year,
        "Capital Expenditures": capex_per_year,
        "Changes in Working Capital": nwc_per_year,
    }

    # List to store lengths of the data
    lengths = [len(val) for val in data.values()]

    ## Attempt to create the DF
    try:
        # Check if all values have the same length
        if len(set(lengths)) > 1:
            raise ValueError("The series have different lengths, cannot merge them.")

        # Convert the dictionary into a DataFrame, with the 'years' as column names
        financial_projections_df = pd.DataFrame(data, index=years)

        # Transpose the DataFrame so the years are the columns
        financial_projections_df = financial_projections_df.transpose()

        # print("DataFrame merged successfully.")

    except ValueError as e:
        sys.exit(f"Error: {e}")

    except KeyError as e:
        sys.exit(f"Missing key in the data: {e}")

    except Exception as e:
        sys.exit(f"An unexpected error occurred: {e}")

    ### UNIT CONVERSION (Millions/Billions) + ROUNDING ###
    financial_projections_df.iloc[:, :] = (
        financial_projections_df.iloc[:, :] * abs_units
    )

    # Convert and round cash_last_period
    cash_last_period = round(float(cash_last_period) * abs_units, 2)

    # Convert and round total_debt
    total_debt = round(float(total_debt) * abs_units, 2)

    # Round df to 2dp
    financial_projections_df = financial_projections_df.round(2)

    ### Ratios of financials into the df ###
    # Add ebit_percentage_of_sales (Operating Income / Total Revenue)
    financial_projections_df.loc["ebit_percentage_of_sales"] = (
        financial_projections_df.apply(
            lambda row: (
                (row["Operating Income (EBIT)"] / row["Total Revenue"]) * 100
            ).round(2)
        )
    )

    # Add taxes_percentage_of_ebit (Provision for Income Taxes / Operating Income)
    financial_projections_df.loc["taxes_percentage_of_ebit"] = (
        financial_projections_df.apply(
            lambda row: (
                (row["Provision for Income Taxes"] / row["Operating Income (EBIT)"])
                * 100
            ).round(2)
        )
    )

    # Depreciation/Depletion
    financial_projections_df.loc["depreciation/depletion_percentage_of_sales"] = (
        financial_projections_df.apply(
            lambda row: (
                (row["Depreciation/Depletion"] / row["Total Revenue"]) * 100
            ).round(2)
        )
    )

    # Capex
    financial_projections_df.loc["capex_percentage_of_sales"] = (
        financial_projections_df.apply(
            lambda row: (
                (row["Capital Expenditures"] / row["Total Revenue"]) * 100
            ).round(2)
        )
    )

    # Change in NWC
    financial_projections_df.loc["changeNWC_percentage_of_sales"] = (
        financial_projections_df.apply(
            lambda row: (
                (row["Changes in Working Capital"] / row["Total Revenue"]) * 100
            ).round(2)
        )
    )

    ## Adding n columns to the df for projections
    for i in range(1, n_periods + 1):
        financial_projections_df[str(i)] = (
            None  # This adds columns named "1", "2", "3", ..., "n"
        )

    ### PROJECTIONS ###
    ## 1) Total Revenue projection
    # Identify year columns (assume columns representing years are 4-digit strings)
    year_columns = [
        col
        for col in financial_projections_df.columns
        if col.isdigit() and len(col) == 4
    ]
    last_year_col = max(year_columns, key=int)  # Get the most recent year column

    # Iterate through the future period columns ('1', '2', ...)
    for i in range(
        1, n_periods + 1
    ):  # n_periods specifies how many future periods to compute
        current_col = str(i)  # Current future period column name
        previous_col = (
            last_year_col if i == 1 else str(i - 1)
        )  # Reference the last year for the first iteration

        # Ensure the current future period column exists in df
        if current_col not in financial_projections_df.columns:
            financial_projections_df[current_col] = (
                None  # Create the column with default None values
            )

        try:
            # Perform the computation only if the previous column value is valid (not NaN or None)
            if pd.notna(financial_projections_df.loc["Total Revenue", previous_col]):
                # Compute and round the total revenue for the current period to 2 decimal places
                financial_projections_df.loc["Total Revenue", current_col] = round(
                    financial_projections_df.loc["Total Revenue", previous_col]
                    * (1 + revenue_growth),
                    2,
                )
            else:
                # Assign None if the previous column has invalid data
                financial_projections_df.loc["Total Revenue", current_col] = None

        except Exception as e:
            print(
                f"Error when trying to compute the future growth for period {current_col}: {e}"
            )

    # Calculate the n-period moving average for all the rows created
    rows_to_update = [
        "ebit_percentage_of_sales",
        "taxes_percentage_of_ebit",
        "depreciation/depletion_percentage_of_sales",
        "capex_percentage_of_sales",
        "changeNWC_percentage_of_sales",
    ]

    for row in rows_to_update:
        fill_moving_average(row, periods=len(years))

    financial_projections_df = financial_projections_df.round(2)

    ### Forecasts based on the %s of each of the categories
    # Define the range of columns to fill
    cols_to_update = financial_projections_df.columns[
        financial_projections_df.columns.get_loc("1") :
    ]  # Adjust range if necessary

    # Loop over each column in the defined range
    for col in cols_to_update:

        # "Operating Income (EBIT)"
        financial_projections_df.loc["Operating Income (EBIT)", col] = round(
            (
                financial_projections_df.loc["ebit_percentage_of_sales", col] / 100
            )  # Divide by 100 to convert percentage to decimal
            * financial_projections_df.loc["Total Revenue", col],
            2,
        )

        # 'Depreciation/Depletion'
        financial_projections_df.loc["Depreciation/Depletion", col] = round(
            (
                financial_projections_df.loc[
                    "depreciation/depletion_percentage_of_sales", col
                ]
                / 100
            )
            * financial_projections_df.loc["Total Revenue", col],
            2,
        )

        # "Capital Expenditures"
        financial_projections_df.loc["Capital Expenditures", col] = round(
            (financial_projections_df.loc["capex_percentage_of_sales", col] / 100)
            * financial_projections_df.loc["Total Revenue", col],
            2,
        )

        # "Changes in Working Capital"
        financial_projections_df.loc["Changes in Working Capital", col] = round(
            (financial_projections_df.loc["changeNWC_percentage_of_sales", col] / 100)
            * financial_projections_df.loc["Total Revenue", col],
            2,
        )

        # "Provision for Income taxes"
        financial_projections_df.loc["Provision for Income Taxes", col] = round(
            financial_projections_df.loc["Operating Income (EBIT)", col]
            * (financial_projections_df.loc["taxes_percentage_of_ebit", col] / 100),
            2,
        )

    ### Extract only the projections of the df
    df_projections = financial_projections_df.loc[:, "1":]

    # Add a new row with column numbers
    df_projections.loc["Future Years"] = list(range(1, len(df_projections.columns) + 1))

    ### 7) EBIAT: Earnings Before Interest and After Taxes ###
    df_projections.loc["EBIAT"] = (
        df_projections.loc["Operating Income (EBIT)"]
        - df_projections.loc["Provision for Income Taxes"]
    )

    ### 8) Unlevered FCF ###
    df_projections.loc["Unlevered FCF"] = (
        df_projections.loc["EBIAT"]
        + df_projections.loc["Depreciation/Depletion"]
        - df_projections.loc["Capital Expenditures"]
        - df_projections.loc["Changes in Working Capital"]
    )

    ### 9) Present Value of FCF (using WACC)
    # We have set the value of the wacc above
    df_projections.loc["PV of FCF"] = (
        df_projections.loc["Unlevered FCF"]
        / (1 + wacc) ** df_projections.loc["Future Years"]
    )

    ### 10) Terminal Value
    # Terminal Value
    terminal_value = round(
        df_projections.loc["Unlevered FCF"].iloc[-1] * ((1 + tgr) / (wacc - tgr)), 2
    )

    # PV of said terminal value
    PV_of_terminal_value = round(
        terminal_value / (1 + wacc) ** df_projections.loc["Future Years"].iloc[-1], 2
    )

    # Enterprise value: Sum of all "PV of FCF" + "PV of terminal value"
    enterprise_value = round(
        PV_of_terminal_value + sum(df_projections.loc["PV of FCF"]), 2
    )

    ### Equity : Enterprise_value + cash - debt
    equity_value = round(
        enterprise_value + round(cash_last_period, 2) - round(total_debt, 2), 2
    )
    print(equity_value)
    try:
        shares_outstanding, current_price = get_shares_outstanding(ticker)
    except ValueError as e:
        print(f"Error: {e}")

    share_price_dcf = equity_value / round(shares_outstanding, 2)

    return current_price, share_price_dcf


if __name__ == "__main__":
    a, b = DCF_pricing_calculator("AAPL", wacc=0.2)
    print((a, b))
