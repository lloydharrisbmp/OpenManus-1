"""
Market Data Analyzer Tool for analyzing financial market data with advanced features.
Supports real-time data fetching, technical analysis, and predictive analytics.
"""

import os
import json
import logging
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from .base import BaseTool

class MarketDataAnalyzerTool(BaseTool):
    """
    Advanced Market Data Analyzer Tool for financial market analysis.
    Features include data fetching, technical analysis, and predictive modeling.
    """

    def __init__(self):
        super().__init__()
        self.name = "MarketDataAnalyzerTool"
        self.description = "Analyzes financial market data with advanced features"
        self.cache_dir = Path("market_data_cache")
        self.output_dir = Path("market_analysis")
        self._setup_directories()
        self._setup_logging()

    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)

    def _setup_logging(self) -> None:
        """Configure logging for the tool."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.name)

    async def fetch_market_data(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> Dict[str, Union[bool, Dict[str, pd.DataFrame]]]:
        """
        Fetch market data for given symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            Dict containing success status and market data
        """
        try:
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')

            market_data = {}
            for symbol in symbols:
                # Check cache first
                cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}_{interval}.csv"
                
                if cache_file.exists():
                    market_data[symbol] = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                else:
                    # Fetch from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date, interval=interval)
                    
                    if data.empty:
                        raise ValueError(f"No data available for symbol {symbol}")
                        
                    # Save to cache
                    data.to_csv(cache_file)
                    market_data[symbol] = data

            return {
                'success': True,
                'data': market_data
            }

        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def analyze_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> Dict[str, Union[bool, pd.DataFrame]]:
        """
        Calculate technical indicators for the given market data.
        
        Args:
            data: Market data DataFrame
            indicators: List of specific indicators to calculate (optional)
            
        Returns:
            Dict containing success status and technical analysis results
        """
        try:
            # Add all technical analysis features
            analysis_df = add_all_ta_features(
                data,
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume"
            )
            
            # Filter specific indicators if requested
            if indicators:
                cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume'] + indicators
                analysis_df = analysis_df[cols_to_keep]

            return {
                'success': True,
                'data': analysis_df
            }

        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def predict_prices(
        self,
        data: pd.DataFrame,
        forecast_days: int = 30,
        features: Optional[List[str]] = None
    ) -> Dict[str, Union[bool, Dict[str, Union[pd.DataFrame, float]]]]:
        """
        Predict future prices using machine learning.
        
        Args:
            data: Market data DataFrame with technical indicators
            forecast_days: Number of days to forecast
            features: List of features to use for prediction
            
        Returns:
            Dict containing success status and prediction results
        """
        try:
            if not features:
                # Use all technical indicators as features
                features = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Volume']]

            # Prepare data
            X = data[features].fillna(0)
            y = data['Close'].shift(-forecast_days).fillna(method='ffill')

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X[:-forecast_days], y[:-forecast_days], test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)

            # Predict future prices
            future_data = X[-forecast_days:].copy()
            future_data_scaled = scaler.transform(future_data)
            future_predictions = model.predict(future_data_scaled)

            # Create forecast DataFrame
            forecast_dates = pd.date_range(
                start=data.index[-1] + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted_Close': future_predictions
            }).set_index('Date')

            return {
                'success': True,
                'predictions': {
                    'forecast': forecast_df,
                    'rmse': rmse,
                    'accuracy_score': model.score(X_test_scaled, y_test)
                }
            }

        except Exception as e:
            self.logger.error(f"Error predicting prices: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def generate_analysis_charts(
        self,
        data: pd.DataFrame,
        analysis_results: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[bool, List[str]]]:
        """
        Generate analysis charts and visualizations.
        
        Args:
            data: Original market data
            analysis_results: Technical analysis results
            predictions: Price predictions (optional)
            
        Returns:
            Dict containing success status and paths to generated charts
        """
        try:
            chart_paths = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Price and Volume Chart
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(data.index, data['Close'], label='Close Price')
            if predictions is not None:
                plt.plot(predictions.index, predictions['Predicted_Close'], 'r--', label='Predicted Price')
            plt.title('Price History and Predictions')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.bar(data.index, data['Volume'], alpha=0.5, label='Volume')
            plt.title('Trading Volume')
            plt.legend()

            chart_path = self.output_dir / "charts" / f"price_volume_{timestamp}.png"
            plt.savefig(chart_path)
            plt.close()
            chart_paths.append(str(chart_path))

            # Technical Indicators Chart
            plt.figure(figsize=(12, 8))
            for i, indicator in enumerate(['momentum_rsi', 'trend_sma_fast', 'volatility_bbm'], 1):
                if indicator in analysis_results.columns:
                    plt.subplot(3, 1, i)
                    plt.plot(analysis_results.index, analysis_results[indicator], label=indicator)
                    plt.title(f'{indicator.replace("_", " ").title()}')
                    plt.legend()

            chart_path = self.output_dir / "charts" / f"technical_indicators_{timestamp}.png"
            plt.savefig(chart_path)
            plt.close()
            chart_paths.append(str(chart_path))

            # Correlation Heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = analysis_results.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Indicator Correlation Matrix')

            chart_path = self.output_dir / "charts" / f"correlation_matrix_{timestamp}.png"
            plt.savefig(chart_path)
            plt.close()
            chart_paths.append(str(chart_path))

            return {
                'success': True,
                'chart_paths': chart_paths
            }

        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def analyze_market_sentiment(
        self,
        symbol: str,
        news_days: int = 30
    ) -> Dict[str, Union[bool, Dict[str, Union[float, List[Dict]]]]]:
        """
        Analyze market sentiment using news and social media data.
        
        Args:
            symbol: Stock symbol
            news_days: Number of days of news to analyze
            
        Returns:
            Dict containing success status and sentiment analysis results
        """
        try:
            # Fetch news data using yfinance
            ticker = yf.Ticker(symbol)
            news_data = ticker.news

            # Process and analyze sentiment (simplified version)
            sentiment_scores = []
            processed_news = []

            for news in news_data:
                # In a real implementation, you would use a proper sentiment analysis model
                # This is a simplified example
                sentiment_score = 0.5  # Neutral sentiment
                
                processed_news.append({
                    'title': news.get('title', ''),
                    'publisher': news.get('publisher', ''),
                    'link': news.get('link', ''),
                    'published': news.get('published', ''),
                    'sentiment': sentiment_score
                })
                
                sentiment_scores.append(sentiment_score)

            average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.5

            return {
                'success': True,
                'sentiment': {
                    'average_score': average_sentiment,
                    'news_items': processed_news
                }
            }

        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def generate_market_report(
        self,
        symbol: str,
        analysis_results: Dict,
        output_format: str = 'html'
    ) -> Dict[str, Union[bool, str]]:
        """
        Generate a comprehensive market analysis report.
        
        Args:
            symbol: Stock symbol
            analysis_results: Dictionary containing all analysis results
            output_format: Output format ('html' or 'pdf')
            
        Returns:
            Dict containing success status and report path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / "reports" / f"market_analysis_{symbol}_{timestamp}.{output_format}"

            # Create report content
            report_content = f"""
            <h1>Market Analysis Report: {symbol}</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Technical Analysis</h2>
            <img src="{analysis_results['charts']['technical_indicators']}" alt="Technical Indicators">
            
            <h2>Price Predictions</h2>
            <p>RMSE: {analysis_results['predictions']['rmse']:.2f}</p>
            <p>Accuracy Score: {analysis_results['predictions']['accuracy_score']:.2%}</p>
            <img src="{analysis_results['charts']['price_volume']}" alt="Price and Volume">
            
            <h2>Market Sentiment</h2>
            <p>Average Sentiment Score: {analysis_results['sentiment']['average_score']:.2f}</p>
            
            <h2>Correlation Analysis</h2>
            <img src="{analysis_results['charts']['correlation_matrix']}" alt="Correlation Matrix">
            """

            # Save report
            with open(report_file, 'w') as f:
                f.write(report_content)

            return {
                'success': True,
                'report_path': str(report_file)
            }

        except Exception as e:
            self.logger.error(f"Error generating market report: {e}")
            return {
                'success': False,
                'error': str(e)
            } 