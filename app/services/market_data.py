"""
Market Data Service for fetching and analyzing financial market data.
"""

import aiohttp
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

@dataclass
class MarketDataResult:
    """Structured result object for market data operations."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    source: str = "market"
    timestamp: datetime = datetime.now()

class MarketDataService:
    """
    Service for fetching and analyzing financial market data.
    Features:
    - Real-time and historical data fetching
    - Multiple data providers support
    - Rate limiting and caching
    - Data validation and cleaning
    - Basic technical analysis
    """

    def __init__(
        self,
        api_keys: Dict[str, str],
        cache_duration: int = 300,  # 5 minutes
        rate_limit: int = 5  # requests per second
    ):
        self.api_keys = api_keys
        self.cache_duration = cache_duration
        self.rate_limit = rate_limit
        self._setup_logging()
        self._setup_rate_limiting()
        self.session = None

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MarketDataService")

    def _setup_rate_limiting(self) -> None:
        """Setup rate limiting."""
        self._request_times = []
        self._rate_limit_lock = asyncio.Lock()

    async def _check_rate_limit(self) -> None:
        """Enforce rate limiting."""
        async with self._rate_limit_lock:
            now = datetime.now()
            # Remove old requests
            self._request_times = [t for t in self._request_times 
                                 if (now - t).total_seconds() < 1]
            
            if len(self._request_times) >= self.rate_limit:
                # Wait until we can make another request
                sleep_time = 1 - (now - self._request_times[0]).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self._request_times.append(now)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_stock_price(
        self,
        symbol: str,
        provider: str = "alpha_vantage"
    ) -> MarketDataResult:
        """
        Get current stock price.
        
        Args:
            symbol: Stock symbol
            provider: Data provider name
            
        Returns:
            MarketDataResult object
        """
        try:
            await self._check_rate_limit()
            
            if provider == "alpha_vantage":
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function=GLOBAL_QUOTE"
                    f"&symbol={symbol}"
                    f"&apikey={self.api_keys['alpha_vantage']}"
                )
                
                session = await self._get_session()
                async with session.get(url) as response:
                    if response.status != 200:
                        return MarketDataResult(
                            success=False,
                            error=f"HTTP {response.status}"
                        )
                    
                    data = await response.json()
                    if "Global Quote" not in data:
                        return MarketDataResult(
                            success=False,
                            error="Invalid response format"
                        )
                    
                    quote = data["Global Quote"]
                    result = {
                        "symbol": symbol,
                        "price": float(quote["05. price"]),
                        "volume": int(quote["06. volume"]),
                        "timestamp": quote["07. latest trading day"]
                    }
                    
                    return MarketDataResult(success=True, data=result)
            
            else:
                return MarketDataResult(
                    success=False,
                    error=f"Unsupported provider: {provider}"
                )

        except Exception as e:
            self.logger.error(f"Error fetching stock price: {e}")
            return MarketDataResult(success=False, error=str(e))

    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime = datetime.now(),
        interval: str = "1d",
        provider: str = "alpha_vantage"
    ) -> MarketDataResult:
        """
        Get historical market data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval (1d, 1h, etc.)
            provider: Data provider name
            
        Returns:
            MarketDataResult object with pandas DataFrame
        """
        try:
            await self._check_rate_limit()
            
            if provider == "alpha_vantage":
                function = "TIME_SERIES_DAILY" if interval == "1d" else "TIME_SERIES_INTRADAY"
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function={function}"
                    f"&symbol={symbol}"
                    f"&apikey={self.api_keys['alpha_vantage']}"
                    "&outputsize=full"
                )
                
                if interval != "1d":
                    url += f"&interval={interval}"
                
                session = await self._get_session()
                async with session.get(url) as response:
                    if response.status != 200:
                        return MarketDataResult(
                            success=False,
                            error=f"HTTP {response.status}"
                        )
                    
                    data = await response.json()
                    time_series_key = (
                        "Time Series (Daily)" if interval == "1d"
                        else f"Time Series ({interval})"
                    )
                    
                    if time_series_key not in data:
                        return MarketDataResult(
                            success=False,
                            error="Invalid response format"
                        )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame.from_dict(
                        data[time_series_key],
                        orient="index"
                    )
                    
                    # Clean column names
                    df.columns = [col.split(". ")[1] for col in df.columns]
                    
                    # Convert types
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    # Filter date range
                    df = df[
                        (df.index >= start_date.strftime("%Y-%m-%d")) &
                        (df.index <= end_date.strftime("%Y-%m-%d"))
                    ]
                    
                    return MarketDataResult(success=True, data=df)
            
            else:
                return MarketDataResult(
                    success=False,
                    error=f"Unsupported provider: {provider}"
                )

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return MarketDataResult(success=False, error=str(e))

    async def get_company_info(
        self,
        symbol: str,
        provider: str = "alpha_vantage"
    ) -> MarketDataResult:
        """
        Get company information.
        
        Args:
            symbol: Stock symbol
            provider: Data provider name
            
        Returns:
            MarketDataResult object
        """
        try:
            await self._check_rate_limit()
            
            if provider == "alpha_vantage":
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function=OVERVIEW"
                    f"&symbol={symbol}"
                    f"&apikey={self.api_keys['alpha_vantage']}"
                )
                
                session = await self._get_session()
                async with session.get(url) as response:
                    if response.status != 200:
                        return MarketDataResult(
                            success=False,
                            error=f"HTTP {response.status}"
                        )
                    
                    data = await response.json()
                    if not data or "Symbol" not in data:
                        return MarketDataResult(
                            success=False,
                            error="Invalid response format"
                        )
                    
                    return MarketDataResult(success=True, data=data)
            
            else:
                return MarketDataResult(
                    success=False,
                    error=f"Unsupported provider: {provider}"
                )

        except Exception as e:
            self.logger.error(f"Error fetching company info: {e}")
            return MarketDataResult(success=False, error=str(e))

    def calculate_technical_indicators(
        self,
        data: pd.DataFrame,
        indicators: List[str]
    ) -> MarketDataResult:
        """
        Calculate technical indicators.
        
        Args:
            data: Price data as pandas DataFrame
            indicators: List of indicators to calculate
            
        Returns:
            MarketDataResult object with indicators added to DataFrame
        """
        try:
            df = data.copy()
            
            for indicator in indicators:
                if indicator == "SMA":
                    # Simple Moving Average
                    df["SMA_20"] = df["close"].rolling(window=20).mean()
                    df["SMA_50"] = df["close"].rolling(window=50).mean()
                    
                elif indicator == "EMA":
                    # Exponential Moving Average
                    df["EMA_20"] = df["close"].ewm(span=20).mean()
                    df["EMA_50"] = df["close"].ewm(span=50).mean()
                    
                elif indicator == "RSI":
                    # Relative Strength Index
                    delta = df["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df["RSI"] = 100 - (100 / (1 + rs))
                    
                elif indicator == "MACD":
                    # Moving Average Convergence Divergence
                    exp1 = df["close"].ewm(span=12).mean()
                    exp2 = df["close"].ewm(span=26).mean()
                    df["MACD"] = exp1 - exp2
                    df["Signal"] = df["MACD"].ewm(span=9).mean()
                    
                elif indicator == "BB":
                    # Bollinger Bands
                    sma = df["close"].rolling(window=20).mean()
                    std = df["close"].rolling(window=20).std()
                    df["BB_upper"] = sma + (std * 2)
                    df["BB_lower"] = sma - (std * 2)
                    
                else:
                    self.logger.warning(f"Unknown indicator: {indicator}")
            
            return MarketDataResult(success=True, data=df)

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return MarketDataResult(success=False, error=str(e))

    def analyze_market_sentiment(
        self,
        data: pd.DataFrame,
        window: int = 14
    ) -> MarketDataResult:
        """
        Analyze market sentiment using price action and volume.
        
        Args:
            data: Price data as pandas DataFrame
            window: Analysis window size
            
        Returns:
            MarketDataResult object with sentiment analysis
        """
        try:
            df = data.copy()
            
            # Calculate price momentum
            df["returns"] = df["close"].pct_change()
            df["momentum"] = df["returns"].rolling(window=window).mean()
            
            # Volume analysis
            df["volume_sma"] = df["volume"].rolling(window=window).mean()
            df["volume_ratio"] = df["volume"] / df["volume_sma"]
            
            # Trend strength
            df["trend"] = np.where(df["momentum"] > 0, 1, -1)
            df["trend_strength"] = abs(df["momentum"]) * df["volume_ratio"]
            
            # Overall sentiment score (-1 to 1)
            df["sentiment_score"] = df["trend"] * df["trend_strength"]
            
            # Categorize sentiment
            df["sentiment"] = pd.cut(
                df["sentiment_score"],
                bins=[-np.inf, -0.5, -0.1, 0.1, 0.5, np.inf],
                labels=["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
            )
            
            latest_sentiment = {
                "score": df["sentiment_score"].iloc[-1],
                "category": df["sentiment"].iloc[-1],
                "momentum": df["momentum"].iloc[-1],
                "volume_ratio": df["volume_ratio"].iloc[-1],
                "trend_strength": df["trend_strength"].iloc[-1]
            }
            
            return MarketDataResult(success=True, data=latest_sentiment)

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return MarketDataResult(success=False, error=str(e))

    async def get_market_news(
        self,
        symbol: str,
        provider: str = "alpha_vantage"
    ) -> MarketDataResult:
        """
        Get market news for a symbol.
        
        Args:
            symbol: Stock symbol
            provider: Data provider name
            
        Returns:
            MarketDataResult object with news items
        """
        try:
            await self._check_rate_limit()
            
            if provider == "alpha_vantage":
                url = (
                    "https://www.alphavantage.co/query"
                    f"?function=NEWS_SENTIMENT"
                    f"&tickers={symbol}"
                    f"&apikey={self.api_keys['alpha_vantage']}"
                )
                
                session = await self._get_session()
                async with session.get(url) as response:
                    if response.status != 200:
                        return MarketDataResult(
                            success=False,
                            error=f"HTTP {response.status}"
                        )
                    
                    data = await response.json()
                    if "feed" not in data:
                        return MarketDataResult(
                            success=False,
                            error="Invalid response format"
                        )
                    
                    return MarketDataResult(success=True, data=data["feed"])
            
            else:
                return MarketDataResult(
                    success=False,
                    error=f"Unsupported provider: {provider}"
                )

        except Exception as e:
            self.logger.error(f"Error fetching market news: {e}")
            return MarketDataResult(success=False, error=str(e))

    def generate_market_report(
        self,
        symbol: str,
        data: pd.DataFrame,
        sentiment: Dict,
        news: List[Dict]
    ) -> MarketDataResult:
        """
        Generate a comprehensive market report.
        
        Args:
            symbol: Stock symbol
            data: Price data DataFrame
            sentiment: Sentiment analysis results
            news: News items
            
        Returns:
            MarketDataResult object with report
        """
        try:
            # Calculate key metrics
            latest_price = data["close"].iloc[-1]
            price_change = data["close"].pct_change().iloc[-1]
            volume = data["volume"].iloc[-1]
            avg_volume = data["volume"].mean()
            
            # Generate report
            report = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price_data": {
                    "latest_price": latest_price,
                    "price_change": price_change,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "volume_ratio": volume / avg_volume
                },
                "technical_analysis": {
                    "sma_20": data["SMA_20"].iloc[-1] if "SMA_20" in data else None,
                    "sma_50": data["SMA_50"].iloc[-1] if "SMA_50" in data else None,
                    "rsi": data["RSI"].iloc[-1] if "RSI" in data else None,
                    "macd": data["MACD"].iloc[-1] if "MACD" in data else None
                },
                "sentiment_analysis": sentiment,
                "recent_news": news[:5] if news else []
            }
            
            return MarketDataResult(success=True, data=report)

        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return MarketDataResult(success=False, error=str(e)) 