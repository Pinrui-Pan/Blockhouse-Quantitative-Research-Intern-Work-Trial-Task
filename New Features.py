import numpy as np

class NewFeatures:
    def __init__(self, data):
        self.data = data

    def add_micro_price(self):
        self.data['Micro_Price'] = (self.data['bid_px_00'] * self.data['ask_sz_00'] 
                                    + self.data['ask_px_00'] * self.data['bid_sz_00']) / (self.data['bid_sz_00'] + self.data['ask_sz_00'])
    # 1. The Micro-Price is a trading metric used to estimate the fair value of a security at a given moment, 
    # factoring in the most immediate liquidity on both the buy and sell sides of the order book. 
    # It is particularly useful in markets where there is a significant amount of bid-ask spread 
    # and provides a more accurate representation of the current market price considering available liquidity.
        
    def add_sequential_trade_direction_change(self):
        self.data['Trade_Direction'] = np.where(self.data['side'] == 'B', 1, np.where(self.data['side'] == 'A', -1, 0))
        self.data['STDC'] = self.data['Trade_Direction'].diff().fillna(0)
    # 2. The Sequential Trade Direction Change (STDC) feature quantifies changes in the direction of trading, 
    # capturing shifts from buying to selling and vice versa. 
    # It is included as it helps identify market sentiment shifts and momentum changes, 
    # enabling traders to anticipate potential reversals or continuations in market trends based on trading behavior.

    def add_ask_bid_transition_ratio(self):
        # Identify transitions
        self.data['Bid_to_Ask'] = ((self.data['side'] == 'A') & (self.data['side'].shift(1) == 'B')).astype(int)
        self.data['Ask_to_Bid'] = ((self.data['side'] == 'B') & (self.data['side'].shift(1) == 'A')).astype(int)
        
        # Calculate a rolling sum for both transition types
        window_size = 30
        self.data['Rolling_Bid_to_Ask'] = self.data['Bid_to_Ask'].rolling(window=window_size).sum()
        self.data['Rolling_Ask_to_Bid'] = self.data['Ask_to_Bid'].rolling(window=window_size).sum()

        # Compute the ratio of these transitions to capture the dynamics between buy and sell pressures
        self.data['ABTR'] = np.where(self.data['Rolling_Ask_to_Bid'] != 0,
                                    self.data['Rolling_Bid_to_Ask'] / self.data['Rolling_Ask_to_Bid'],
                                    np.nan)
    # 3. Ask-Bid Transition Ratio (ABTR) measures the frequency and magnitude of transitions between 
    # ask-initiated and bid-initiated trades. Including this feature is beneficial for 
    # analyzing the balance of buying and selling pressure, providing insights into 
    # which side of the market—buyers or sellers—is dominating the trading dynamics at any given time.

    def add_price_recovery_post_trade(self):
        self.data['Deviation_Micro_Price'] = self.data['price'] - self.data['Micro_Price']
        self.data['Recovery_Micro_Price'] = False

        # Define recovery: condition for price coming back within a small percentage of the Micro-Price
        recovery_threshold = 0.01  # 1% of the Micro-Price
        self.data.loc[np.abs(self.data['Deviation_Micro_Price']) <= np.abs(recovery_threshold * self.data['Micro_Price']), 
                      'Recovery_Micro_Price'] = True

        # Calculate time to recovery for Micro-Price
        self.data['Time_To_Recovery_Micro_Price'] = np.nan
        recovery_indices_micro_price = self.data[self.data['Recovery_Micro_Price']].index

        for idx in recovery_indices_micro_price:
            prior_idx = idx - 1
            while prior_idx >= 0 and not self.data.at[prior_idx, 'Recovery_Micro_Price']:
                prior_idx -= 1
            if prior_idx >= 0:
                self.data.at[idx, 'Time_To_Recovery_Micro_Price'] = (self.data.at[idx, 'ts_event'] - self.data.at[prior_idx, 'ts_event']) * 1e-9
    # 4. Price Recovery Post-Trade measures how quickly and closely prices return to a predetermined benchmark level 
    # (such as the micro price in this case) following significant price movements due to trades. 
    # This feature is valuable as it provides insights into the market's liquidity and resilience, 
    # indicating how rapidly prices stabilize after fluctuations, which is critical for assessing market stability and the effectiveness of trading strategies.

    def add_trade_frequency(self):
        self.data['Trade_Frequency'] = self.data['ts_event'].diff().fillna(0).apply(lambda x: 1e9 / x if x > 0 else 0)
    # 5. The Trade Frequency feature measures the frequency of trades by calculating the inverse of the time difference 
    # between consecutive trades, converted to trades per second. It helps identify periods of high or low trading activity, 
    # providing insights into market liquidity and volatility.
    
    def add_all_new_features(self):
        self.add_micro_price()
        self.add_sequential_trade_direction_change()
        self.add_ask_bid_transition_ratio()
        self.add_price_recovery_post_trade()
        self.add_intra_day_time_decay()
        return self.data
