import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# ==========================================
# HELPER FUNCTIONS FOR VOLUME ANALYSIS
# ==========================================

def calculate_relative_volume(current_volume, volume_array, lookback=20):
    """
    Calculate relative volume ratio: current_volume / average_volume
    Returns: ratio (1.0 = average, >1.0 = above average, <1.0 = below average)
    """
    if len(volume_array) < lookback:
        return 0
    adv = np.mean(volume_array[-lookback:])
    return current_volume / adv if adv > 0 else 0

def detect_volume_divergence(df, period=5):
    """
    Detect volume divergence: Price moving but volume not confirming
    Returns: divergence warning string or 'CONFIRMED'
    """
    if len(df) < period * 2:
        return 'NONE'
    
    current = df.iloc[-1]
    prior = df.iloc[-period-1]
    
    # Calculate price direction
    price_change = current['Close'] - prior['Close']
    price_direction = 'UP' if price_change > 0 else 'DOWN'
    
    # Calculate volume trend
    vol_recent = df['Volume'].iloc[-period:].mean()
    vol_prior = df['Volume'].iloc[-period*2:-period].mean()
    vol_change = vol_recent - vol_prior
    vol_direction = 'UP' if vol_change > 0 else 'DOWN'
    vol_trend_pct = ((vol_change / vol_prior) * 100) if vol_prior > 0 else 0
    
    # Check for divergence
    if price_direction != vol_direction:
        severity = 'WARNING' if abs(vol_trend_pct) > 10 else 'MINOR'
        return f'DIVERGENCE_{severity}'
    return 'CONFIRMED'

# ==========================================
# MOMENTUM-BASED STOCK ANALYZER
# Combines Turtle Strategy with Multi-Timeframe Momentum Detection
# ==========================================

class MomentumAnalyzer:
    """
    Identifies stocks with SUSTAINED MOMENTUM across multiple timeframes.
    Uses accelerating momentum, trend alignment, and volatility context.
    """
    
    def __init__(self, tickers, output_dir="momentum_signals", min_confidence=60):
        self.tickers = tickers
        self.results = []
        self.output_dir = output_dir
        self.min_confidence = min_confidence
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # ==========================================
    # 1. MULTI-TIMEFRAME MOMENTUM INDICATORS
    # ==========================================
    
    def calculate_momentum_indicators(self, df):
        """Calculate momentum metrics across multiple timeframes."""
        
        # SHORT-TERM MOMENTUM (5-14 days)
        df['ROC_5'] = df['Close'].pct_change(5) * 100
        df['ROC_14'] = df['Close'].pct_change(14) * 100
        
        # MID-TERM MOMENTUM (20-50 days)
        df['ROC_20'] = df['Close'].pct_change(20) * 100
        df['ROC_50'] = df['Close'].pct_change(50) * 100
        
        # MOMENTUM ACCELERATION (is momentum getting stronger?)
        df['Momentum_Accel'] = df['ROC_14'].diff()  # Change in momentum
        
        # RSI (Relative Strength Index) - multiple timeframes
        df['RSI_7'] = self._calculate_rsi(df['Close'], 7)
        df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
        df['RSI_21'] = self._calculate_rsi(df['Close'], 21)
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # MACD Momentum (is MACD getting stronger?)
        df['MACD_Momentum'] = df['MACD_Histogram'].diff()
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI efficiently."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # ==========================================
    # 2. MOVING AVERAGE TRENDLINES
    # ==========================================
    
    def calculate_moving_averages(self, df):
        """Calculate EMAs and SMAs for trend following."""
        
        # Short-term EMAs (from weeklyfilter_gemini)
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Long-term trend (from turtle_short)
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # TREND CLASSIFICATION
        df['Trend'] = 'NEUTRAL'
        df.loc[df['Close'] > df['EMA_9'], 'Trend'] = 'STRONG_UP'
        df.loc[(df['Close'] <= df['EMA_9']) & (df['Close'] > df['EMA_20']), 'Trend'] = 'UP'
        df.loc[(df['Close'] <= df['EMA_20']) & (df['Close'] > df['EMA_50']), 'Trend'] = 'WEAK_UP'
        df.loc[df['Close'] < df['EMA_50'], 'Trend'] = 'DOWN'
        
        return df
    
    # ==========================================
    # 3. VOLUME & VOLATILITY ANALYSIS
    # ==========================================
    
    def calculate_volume_volatility(self, df):
        """Analyze volume and volatility for confirmation."""
        
        # Volume metrics
        df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA_20']
        
        # RELATIVE VOLUME: Current volume vs 20-day average
        df['Relative_Volume'] = df.apply(
            lambda row: calculate_relative_volume(row['Volume'], df['Volume'].values, lookback=20),
            axis=1
        )
        
        # VOLUME DIVERGENCE: Detect price-volume mismatch
        df['Vol_Divergence'] = 'NONE'
        for i in range(10, len(df)):
            df.loc[df.index[i], 'Vol_Divergence'] = detect_volume_divergence(df.iloc[i-9:i+1], period=5)
        
        # Historical volatility
        df['Returns'] = df['Close'].pct_change()
        df['HV_20'] = df['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        df['HV_50'] = df['Returns'].rolling(window=50).std() * np.sqrt(252) * 100
        
        # ATR (Average True Range)
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        return df
    
    # ==========================================
    # 4. SUSTAINED MOMENTUM DETECTION
    # ==========================================
    
    def detect_sustained_momentum(self, df):
        """
        Identifies SUSTAINED MOMENTUM - momentum that persists and accelerates.
        Returns dataframe with momentum flags and scores.
        """
        
        current = df.iloc[-1]
        yesterday = df.iloc[-2]
        
        momentum_signals = {}
        
        # === UPSIDE MOMENTUM SIGNALS ===
        
        # 1. Multi-timeframe momentum alignment (all positive)
        multi_tf_aligned = (current['ROC_5'] > 0 and 
                          current['ROC_14'] > 0 and 
                          current['ROC_20'] > 0)
        momentum_signals['Multi_TF_Aligned'] = multi_tf_aligned
        
        # 2. Momentum acceleration (getting stronger)
        mom_accelerating = current['Momentum_Accel'] > 0 and current['ROC_14'] > yesterday['ROC_14']
        momentum_signals['Momentum_Accelerating'] = mom_accelerating
        
        # 3. EMA alignment (price above all key EMAs = strong uptrend)
        ema_aligned_up = (current['Close'] > current['EMA_9'] and 
                         current['EMA_9'] > current['EMA_20'] and 
                         current['EMA_20'] > current['EMA_50'])
        momentum_signals['EMA_Aligned_Up'] = ema_aligned_up
        
        # 4. MACD positive momentum
        macd_positive = current['MACD_Histogram'] > 0 and current['MACD_Momentum'] > 0
        momentum_signals['MACD_Positive'] = macd_positive
        
        # 5. RSI confirming (7 and 14 in productive zones, not overbought)
        rsi_confirming = (30 < current['RSI_7'] < 85 and 
                         30 < current['RSI_14'] < 80)
        momentum_signals['RSI_Confirming'] = rsi_confirming
        
        # 6. Volume confirmation (volume increasing)
        vol_confirming = current['Vol_Ratio'] > 1.0
        momentum_signals['Volume_Confirming'] = vol_confirming
        
        # 6b. Relative volume gate (>= 1.0 required)
        rel_vol_gate = current['Relative_Volume'] >= 1.0
        momentum_signals['Relative_Vol_Gate'] = rel_vol_gate
        
        # 6c. No volume divergence warning
        no_divergence = 'DIVERGENCE' not in str(current.get('Vol_Divergence', 'NONE'))
        momentum_signals['Volume_Confirmed'] = no_divergence
        momentum_signals['Vol_Divergence'] = current.get('Vol_Divergence', 'NONE')
        
        # 7. Distance to EMA_50 (in strong uptrend, price should be 2%+ above EMA50)
        distance_to_ema50 = ((current['Close'] - current['EMA_50']) / current['EMA_50']) * 100
        momentum_signals['Distance_EMA50'] = distance_to_ema50
        far_from_support = distance_to_ema50 > 2.0
        momentum_signals['Far_From_Support'] = far_from_support
        
        # === DOWNSIDE MOMENTUM SIGNALS ===
        
        # 1. Multi-timeframe negative momentum
        multi_tf_down = (current['ROC_5'] < 0 and 
                        current['ROC_14'] < 0 and 
                        current['ROC_20'] < 0)
        momentum_signals['Multi_TF_Down'] = multi_tf_down
        
        # 2. Momentum accelerating downside
        mom_accel_down = current['Momentum_Accel'] < 0 and current['ROC_14'] < yesterday['ROC_14']
        momentum_signals['Momentum_Accel_Down'] = mom_accel_down
        
        # 3. EMA alignment (bearish)
        ema_aligned_down = (current['Close'] < current['EMA_50'] and 
                           current['EMA_50'] < current['EMA_20'] and 
                           current['EMA_20'] < current['EMA_9'])
        momentum_signals['EMA_Aligned_Down'] = ema_aligned_down
        
        # 4. MACD negative momentum
        macd_negative = current['MACD_Histogram'] < 0 and current['MACD_Momentum'] < 0
        momentum_signals['MACD_Negative'] = macd_negative
        
        return momentum_signals
    
    # ==========================================
    # 5. MOMENTUM SCORE & CONFIDENCE
    # ==========================================
    
    def calculate_momentum_score(self, df, ticker):
        """Calculate comprehensive momentum confidence score (0-100)."""
        
        momentum_sigs = self.detect_sustained_momentum(df)
        current = df.iloc[-1]
        
        score = 0
        max_score = 100
        
        # === BULLISH ASSESSMENT (0-100) ===
        
        if momentum_sigs['Multi_TF_Aligned']:
            score += 15  # Multi-timeframe alignment is strong signal
        
        if momentum_sigs['Momentum_Accelerating']:
            score += 15  # Accelerating momentum is critical
        
        if momentum_sigs['EMA_Aligned_Up']:
            score += 20  # Perfect EMA alignment = strong trend
        
        if momentum_sigs['MACD_Positive']:
            score += 12  # MACD confirmation
        
        if momentum_sigs['RSI_Confirming']:
            score += 12  # RSI in productive zone
        
        if momentum_sigs['Volume_Confirming']:
            score += 10  # Volume backing the move
        
        if momentum_sigs['Far_From_Support']:
            score += 8   # Strong move, not near support levels
        
        # Bonus: Strong ROC values
        if current['ROC_14'] > 10:
            score += 5   # Strong momentum reading
        elif current['ROC_14'] > 5:
            score += 2
        
        # --- BEARISH ASSESSMENT ---
        
        # Penalty if signals are contradictory
        if momentum_sigs['EMA_Aligned_Down']:
            score -= 30
        
        if momentum_sigs['Multi_TF_Down']:
            score -= 20
        
        if momentum_sigs['MACD_Negative']:
            score -= 15
        
        if current['RSI_14'] > 85:  # Overbought
            score -= 10
        
        # Cap the score
        score = max(0, min(100, score))
        
        return score, momentum_sigs
    
    # ==========================================
    # 6. DAY-OF-WEEK SPECIAL ANALYSIS
    # ==========================================
    
    def get_weekly_bias(self, ticker_data):
        """Apply day-of-week optimizations inspired by weeklyfilter_gemini."""
        
        today = datetime.now().weekday()  # 0=Mon, 1=Tue, ..., 4=Fri
        current = ticker_data.iloc[-1]
        
        day_names = {0: 'MONDAY', 1: 'TUESDAY', 2: 'WEDNESDAY', 3: 'THURSDAY', 4: 'FRIDAY'}
        day_name = day_names.get(today, 'WEEKEND')
        
        bias = {
            'day': day_name,
            'trading_tendency': '',
            'caution_flags': []
        }
        
        # Gap analysis
        prev_close = ticker_data.iloc[-2]['Close']
        gap_pct = ((current['Open'] - prev_close) / prev_close) * 100
        
        if today == 0:  # MONDAY
            bias['trading_tendency'] = 'Look for gap continuation or reversal'
            if abs(gap_pct) < 0.5:
                bias['caution_flags'].append('Narrow gap - lower follow-through')
        
        elif today == 1:  # TUESDAY
            bias['trading_tendency'] = 'Trend confirmation day'
            if current['Volume'] < ticker_data.iloc[-20:-1]['Volume'].mean():
                bias['caution_flags'].append('Below-average volume - weak confirmation')
        
        elif today == 2:  # WEDNESDAY
            bias['trading_tendency'] = 'Mid-week pivot point'
            if 40 < current['RSI_14'] < 60:
                bias['trading_tendency'] = 'Neutral range - use support/resistance'
        
        elif today == 3:  # THURSDAY
            bias['trading_tendency'] = 'Momentum acceleration into Friday'
            if current['ROC_14'] > 5:
                bias['trading_tendency'] = 'Strong momentum - likely to continue'
        
        elif today == 4:  # FRIDAY
            bias['trading_tendency'] = 'Scout for next week (monitor close position)'
            close_pos = (current['Close'] - current['Low']) / (current['High'] - current['Low']) if current['High'] != current['Low'] else 0.5
            if close_pos > 0.75:
                bias['trading_tendency'] = 'Strong close - potential Monday gap up'
        
        return bias
    
    # ==========================================
    # 7. MAIN SCANNER
    # ==========================================
    
    def run(self):
        """Scan universe for sustained momentum."""
        
        print(f"🔍 MOMENTUM ANALYZER - Scanning {len(self.tickers)} tickers...")
        print(f"📊 Minimum Confidence Threshold: {self.min_confidence}%")
        print("-" * 80)
        
        for ticker in self.tickers:
            try:
                self.process_ticker(ticker)
            except Exception as e:
                print(f"⚠️  Error processing {ticker}: {e}")
        
        return pd.DataFrame(self.results)
    
    def process_ticker(self, ticker):
        """Analyze single ticker for sustained momentum."""
        
        # Fetch 1 year of data for robust indicators
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=False)
        
        if len(df) < 100:
            return  # Not enough data
        
        # Clean MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Calculate all indicators
        df = self.calculate_momentum_indicators(df)
        df = self.calculate_moving_averages(df)
        df = self.calculate_volume_volatility(df)
        
        # Get current state
        current = df.iloc[-1]
        momentum_score, momentum_sigs = self.calculate_momentum_score(df, ticker)
        weekly_bias = self.get_weekly_bias(df)
        
        # Determine signal direction
        signal = "NONE"
        signal_direction = None
        
        # BULLISH signal: Strong upside momentum
        if (momentum_sigs['Multi_TF_Aligned'] and 
            momentum_sigs['Momentum_Accelerating'] and 
            momentum_sigs['EMA_Aligned_Up'] and 
            momentum_score >= self.min_confidence):
            signal = "STRONG_UPSIDE_MOMENTUM"
            signal_direction = "BULLISH"
        
        # BEARISH signal: Strong downside momentum
        elif (momentum_sigs['Multi_TF_Down'] and 
              momentum_sigs['Momentum_Accel_Down'] and 
              momentum_sigs['EMA_Aligned_Down'] and 
              momentum_score >= self.min_confidence):
            signal = "STRONG_DOWNSIDE_MOMENTUM"
            signal_direction = "BEARISH"
        
        # MODERATE UPSIDE
        elif (momentum_sigs['Multi_TF_Aligned'] and 
              momentum_sigs['MACD_Positive'] and 
              current['Trend'] in ['STRONG_UP', 'UP'] and 
              momentum_score >= 50):
            signal = "MODERATE_UPSIDE_MOMENTUM"
            signal_direction = "BULLISH"
        
        # MODERATE DOWNSIDE
        elif (momentum_sigs['Multi_TF_Down'] and 
              momentum_sigs['MACD_Negative'] and 
              current['Trend'] in ['DOWN'] and 
              momentum_score >= 50):
            signal = "MODERATE_DOWNSIDE_MOMENTUM"
            signal_direction = "BEARISH"
        
        # Only store signals above threshold
        if signal != "NONE":
            self.results.append({
                'Ticker': ticker,
                'Signal': signal,
                'Direction': signal_direction,
                'Confidence': round(momentum_score, 1),
                'Current_Price': round(current['Close'], 2),
                'Trend': current['Trend'],
                
                # Momentum metrics
                'ROC_5': round(current['ROC_5'], 2),
                'ROC_14': round(current['ROC_14'], 2),
                'ROC_20': round(current['ROC_20'], 2),
                'Momentum_Accel': round(current['Momentum_Accel'], 2),
                
                # Technical indicators
                'RSI_7': round(current['RSI_7'], 1),
                'RSI_14': round(current['RSI_14'], 1),
                'MACD_Histogram': round(current['MACD_Histogram'], 4),
                
                # Volume & Volatility
                'Volume_Ratio': round(current['Vol_Ratio'], 2),
                'HV_20': round(current['HV_20'], 2),
                'ATR': round(current['ATR'], 2),
                
                # Position metrics
                'Distance_EMA50_Pct': round(momentum_sigs['Distance_EMA50'], 2),
                'EMA_9': round(current['EMA_9'], 2),
                'EMA_20': round(current['EMA_20'], 2),
                'EMA_50': round(current['EMA_50'], 2),
                
                # Weekly context
                'Day_of_Week': weekly_bias['day'],
                'Weekly_Bias': weekly_bias['trading_tendency'],
            })
    
    # ==========================================
    # 8. REPORTING & VISUALIZATION
    # ==========================================
    
    def save_results(self, results):
        """Save comprehensive momentum analysis report."""
        
        if results.empty:
            print("\n⚠️  No momentum signals detected meeting confidence threshold.")
            return
        
        # Add metadata
        results['Timestamp'] = self.timestamp
        results['Date'] = self.date_str
        
        # 1. Save to CSV
        csv_path = os.path.join(self.output_dir, f"momentum_{self.date_str}.csv")
        results.to_csv(csv_path, index=False)
        print(f"\n✓ Momentum results saved to: {csv_path}")
        
        # 2. Save to JSON
        json_path = os.path.join(self.output_dir, f"momentum_{self.date_str}.json")
        results.to_json(json_path, orient="records", indent=2)
        print(f"✓ JSON results saved to: {json_path}")
        
        # 3. Create detailed report
        self.create_detailed_report(results)
    
    def create_detailed_report(self, results):
        """Generate comprehensive momentum analysis report."""
        
        report_path = os.path.join(self.output_dir, f"momentum_report_{self.date_str}.txt")
        
        bullish = results[results['Direction'] == 'BULLISH'].sort_values('Confidence', ascending=False)
        bearish = results[results['Direction'] == 'BEARISH'].sort_values('Confidence', ascending=False)
        
        with open(report_path, "w") as f:
            f.write("=" * 100 + "\n")
            f.write("     SUSTAINED MOMENTUM ANALYZER - MULTI-TIMEFRAME MOMENTUM DETECTION\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"Analysis Date/Time: {self.timestamp}\n")
            f.write(f"Tickers Analyzed:   {len(results)}\n")
            f.write(f"Confidence Filter:  {self.min_confidence}%\n\n")
            
            f.write("-" * 100 + "\n")
            f.write("METHODOLOGY\n")
            f.write("-" * 100 + "\n")
            f.write("This analyzer identifies stocks with SUSTAINED MOMENTUM - momentum that accelerates\n")
            f.write("across multiple timeframes and is confirmed by technical indicators.\n\n")
            
            f.write("KEY METRICS:\n")
            f.write("  • Multi-Timeframe ROC: ROC-5, ROC-14, ROC-20 alignment\n")
            f.write("  • Momentum Acceleration: Change in momentum to detect acceleration\n")
            f.write("  • EMA Alignment: Price > EMA-9 > EMA-20 > EMA-50 (bullish) or reverse (bearish)\n")
            f.write("  • MACD Momentum: MACD histogram and its momentum\n")
            f.write("  • RSI Confirmation: RSI-7 and RSI-14 in productive zones\n")
            f.write("  • Volume Confirmation: Volume above 20-day average\n\n")
            
            f.write("-" * 100 + "\n")
            f.write(f"BULLISH MOMENTUM ({len(bullish)} stocks)\n")
            f.write("-" * 100 + "\n\n")
            
            if not bullish.empty:
                for idx, row in bullish.iterrows():
                    f.write(f"{'='*50}\n")
                    f.write(f"Ticker:             {row['Ticker']}\n")
                    f.write(f"Signal:             {row['Signal']}\n")
                    f.write(f"Confidence:         {row['Confidence']}%\n")
                    f.write(f"Current Price:      ${row['Current_Price']:.2f}\n")
                    f.write(f"Trend:              {row['Trend']}\n\n")
                    
                    f.write(f"MOMENTUM METRICS:\n")
                    f.write(f"  ROC-5:              {row['ROC_5']:>7.2f}%\n")
                    f.write(f"  ROC-14:             {row['ROC_14']:>7.2f}%\n")
                    f.write(f"  ROC-20:             {row['ROC_20']:>7.2f}%\n")
                    f.write(f"  Momentum Accel:     {row['Momentum_Accel']:>7.2f}\n\n")
                    
                    f.write(f"TECHNICAL INDICATORS:\n")
                    f.write(f"  RSI-7/14:           {row['RSI_7']:.1f} / {row['RSI_14']:.1f}\n")
                    f.write(f"  MACD Histogram:     {row['MACD_Histogram']:.4f}\n")
                    f.write(f"  Volume Ratio:       {row['Volume_Ratio']:.2f}x (avg)\n")
                    f.write(f"  HV-20:              {row['HV_20']:.2f}%\n")
                    f.write(f"  ATR:                ${row['ATR']:.2f}\n\n")
                    
                    f.write(f"MOVING AVERAGES:\n")
                    f.write(f"  EMA-9:              ${row['EMA_9']:.2f}\n")
                    f.write(f"  EMA-20:             ${row['EMA_20']:.2f}\n")
                    f.write(f"  EMA-50:             ${row['EMA_50']:.2f}\n")
                    f.write(f"  Distance to EMA-50: {row['Distance_EMA50_Pct']:.2f}%\n\n")
                    
                    f.write(f"WEEKLY CONTEXT:\n")
                    f.write(f"  Day of Week:        {row['Day_of_Week']}\n")
                    f.write(f"  Weekly Bias:        {row['Weekly_Bias']}\n\n")
            else:
                f.write("No bullish momentum signals detected.\n\n")
            
            f.write("-" * 100 + "\n")
            f.write(f"BEARISH MOMENTUM ({len(bearish)} stocks)\n")
            f.write("-" * 100 + "\n\n")
            
            if not bearish.empty:
                for idx, row in bearish.iterrows():
                    f.write(f"{'='*50}\n")
                    f.write(f"Ticker:             {row['Ticker']}\n")
                    f.write(f"Signal:             {row['Signal']}\n")
                    f.write(f"Confidence:         {row['Confidence']}%\n")
                    f.write(f"Current Price:      ${row['Current_Price']:.2f}\n")
                    f.write(f"Trend:              {row['Trend']}\n\n")
                    
                    f.write(f"MOMENTUM METRICS:\n")
                    f.write(f"  ROC-5:              {row['ROC_5']:>7.2f}%\n")
                    f.write(f"  ROC-14:             {row['ROC_14']:>7.2f}%\n")
                    f.write(f"  ROC-20:             {row['ROC_20']:>7.2f}%\n")
                    f.write(f"  Momentum Accel:     {row['Momentum_Accel']:>7.2f}\n\n")
                    
                    f.write(f"TECHNICAL INDICATORS:\n")
                    f.write(f"  RSI-7/14:           {row['RSI_7']:.1f} / {row['RSI_14']:.1f}\n")
                    f.write(f"  MACD Histogram:     {row['MACD_Histogram']:.4f}\n")
                    f.write(f"  Volume Ratio:       {row['Volume_Ratio']:.2f}x (avg)\n")
                    f.write(f"  HV-20:              {row['HV_20']:.2f}%\n")
                    f.write(f"  ATR:                ${row['ATR']:.2f}\n\n")
                    
                    f.write(f"MOVING AVERAGES:\n")
                    f.write(f"  EMA-9:              ${row['EMA_9']:.2f}\n")
                    f.write(f"  EMA-20:             ${row['EMA_20']:.2f}\n")
                    f.write(f"  EMA-50:             ${row['EMA_50']:.2f}\n")
                    f.write(f"  Distance to EMA-50: {row['Distance_EMA50_Pct']:.2f}%\n\n")
                    
                    f.write(f"WEEKLY CONTEXT:\n")
                    f.write(f"  Day of Week:        {row['Day_of_Week']}\n")
                    f.write(f"  Weekly Bias:        {row['Weekly_Bias']}\n\n")
            else:
                f.write("No bearish momentum signals detected.\n\n")
        
        print(f"✓ Detailed report saved to: {report_path}")


# ==========================================
# 9. COLORIZED CONSOLE OUTPUT
# ==========================================

def print_momentum_results(results):
    """Display momentum analysis results with color coding."""
    
    # ANSI Colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    print(f"\n{BOLD}{'='*90} SUSTAINED MOMENTUM ANALYZER {'='*90}{END}\n")
    
    if results.empty or 'Signal' not in results.columns:
        print(f"{YELLOW}No momentum signals detected for stocks analyzed.{END}\n")
        return
    
    # Separate by direction
    bullish = results[results['Direction'] == 'BULLISH'].sort_values('Confidence', ascending=False)
    bearish = results[results['Direction'] == 'BEARISH'].sort_values('Confidence', ascending=False)
    
    # BULLISH SIGNALS
    if not bullish.empty:
        print(f"\n{BOLD}{GREEN}📈 BULLISH MOMENTUM ({len(bullish)} stocks){END}\n")
        
        for _, row in bullish.iterrows():
            conf_color = GREEN if row['Confidence'] >= 75 else YELLOW
            roc_color = GREEN if row['ROC_14'] > 0 else RED
            
            print(f"{BOLD}{GREEN}▲ {row['Ticker']:<8}{END} @ ${row['Current_Price']:<7.2f}  "
                  f"{conf_color}Confidence: {row['Confidence']:.0f}%{END}")
            print(f"   Signal:   {row['Signal']}")
            print(f"   Trend:    {row['Trend']:<15} ROC-14: {roc_color}{row['ROC_14']:>6.2f}%{END}  "
                  f"RSI-14: {row['RSI_14']:.0f}")
            print(f"   Volume:   {row['Volume_Ratio']:.2f}x average   Above EMA-50 by {row['Distance_EMA50_Pct']:.1f}%")
            print()
    
    # BEARISH SIGNALS
    if not bearish.empty:
        print(f"\n{BOLD}{RED}📉 BEARISH MOMENTUM ({len(bearish)} stocks){END}\n")
        
        for _, row in bearish.iterrows():
            conf_color = RED if row['Confidence'] >= 75 else YELLOW
            roc_color = RED if row['ROC_14'] < 0 else GREEN
            
            print(f"{BOLD}{RED}▼ {row['Ticker']:<8}{END} @ ${row['Current_Price']:<7.2f}  "
                  f"{conf_color}Confidence: {row['Confidence']:.0f}%{END}")
            print(f"   Signal:   {row['Signal']}")
            print(f"   Trend:    {row['Trend']:<15} ROC-14: {roc_color}{row['ROC_14']:>6.2f}%{END}  "
                  f"RSI-14: {row['RSI_14']:.0f}")
            print(f"   Volume:   {row['Volume_Ratio']:.2f}x average   Below EMA-50 by {abs(row['Distance_EMA50_Pct']):.1f}%")
            print()
    
    print(f"{BOLD}{'='*90}{END}\n")
    
    # Summary
    avg_conf_bullish = bullish['Confidence'].mean() if not bullish.empty else 0
    avg_conf_bearish = bearish['Confidence'].mean() if not bearish.empty else 0
    
    print(f"{CYAN}SUMMARY:{END}")
    if not bullish.empty:
        print(f"  {GREEN}Bullish avg confidence: {avg_conf_bullish:.0f}%{END}")
    if not bearish.empty:
        print(f"  {RED}Bearish avg confidence: {avg_conf_bearish:.0f}%{END}")
    print()


# ==========================================
# 10. EXECUTION
# ==========================================

if __name__ == "__main__":
    # Example ticker universe (mix of stocks with different momentum profiles)
    tickers = [
         'META','GOOG','MU', 'NVDA', 'NFLX', 'MSFT', 'AAPL', 'AMD', 'AMZN',
        'SNDK', 'CRWV', 'BLSH', 'TSLA', 'ONDS', 'RKLB', 'NBIS', 'COIN', 'MRVL', 'META',
        'SPY', 'QQQ', 'IWM', 'COHR', 'LITE', 'SNPS', 'CLS', 'ASTS', 'LUNR', 'CDNS', 'STX',
    ]
    
    # Initialize analyzer with confidence threshold
    # min_confidence=60: Filters out weak momentum signals
    analyzer = MomentumAnalyzer(tickers, min_confidence=60)
    
    # Run scan
    results = analyzer.run()
    
    # Display results
    print_momentum_results(results)
    
    # Save results
    if not results.empty:
        analyzer.save_results(results)
    else:
        print("[INFO] No sustained momentum signals detected this scan.")
