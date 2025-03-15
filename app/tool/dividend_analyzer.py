from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Callable, Any
import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
# Handle missing statsmodels with a try/except
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:
    # Fallback to a simple moving average approach if statsmodels is not available
    class ExponentialSmoothing:
        def __init__(self, *args, **kwargs):
            self.data = args[0] if args else None
            
        def fit(self):
            return self
            
        def forecast(self, steps):
            if self.data is None or len(self.data) == 0:
                return pd.Series([0] * steps)
            # Simple moving average as fallback
            return pd.Series([self.data.mean()] * steps)

from app.tool.base import BaseTool

class DividendAnalyzerParameters(BaseModel):
    stock_code: str = Field(..., description="ASX stock code (e.g., 'CBA.AX')")
    start_date: Optional[str] = Field(
        default=None, 
        description="Start date for analysis (YYYY-MM-DD)"
    )
    forecast_periods: int = Field(
        default=4, 
        description="Number of periods to forecast"
    )
    output_dir: str = Field(
        default="client_documents/reports",
        description="Directory to save generated reports"
    )

class DividendAnalyzerTool(BaseTool):
    """Tool for fetching and analyzing dividend data for stocks."""
    
    name: str = "dividend_analyzer"
    description: str = "Fetches and analyzes dividend data for specific stocks"
    parameters: Dict[str, Any] = {
        "stock_code": Field(..., description="ASX stock code (e.g., 'CBA.AX')"),
        "start_date": Field(
            default=None, 
            description="Start date for analysis (YYYY-MM-DD)"
        ),
        "forecast_periods": Field(
            default=4, 
            description="Number of periods to forecast"
        ),
        "output_dir": Field(
            default="client_documents/reports",
            description="Directory to save generated reports"
        )
    }

    def __init__(self):
        super().__init__()
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    def _update_progress(self, message: str, percentage: float):
        """Update progress if callback is set"""
        if self.progress_callback and callable(self.progress_callback):
            try:
                self.progress_callback({"message": message, "percentage": percentage})
            except Exception:
                # Safely ignore errors in the callback
                pass

    def _fetch_dividend_data(self, stock_code: str, start_date: Optional[str]) -> pd.DataFrame:
        try:
            stock = yf.Ticker(stock_code)
            dividends = stock.dividends
            if start_date:
                dividends = dividends[dividends.index >= pd.Timestamp(start_date)]
            return dividends.to_frame()
        except Exception as e:
            raise ValueError(f"Error fetching dividend data: {str(e)}")

    def _calculate_metrics(self, dividends_df: pd.DataFrame) -> Dict:
        """Calculate key dividend metrics"""
        try:
            # Calculate year-over-year growth rates
            annual_dividends = dividends_df.resample('Y').sum()
            growth_rates = annual_dividends.pct_change().dropna()
            
            metrics = {
                "avg_growth_rate": growth_rates.mean(),
                "latest_dividend": dividends_df.iloc[-1],
                "total_dividends": dividends_df.sum(),
                "dividend_frequency": len(dividends_df) / (
                    (dividends_df.index[-1] - dividends_df.index[0]).days / 365.25
                )
            }
            return metrics
        except Exception as e:
            raise ValueError(f"Error calculating metrics: {str(e)}")

    def _forecast_dividends(self, historical_data: pd.Series, periods: int) -> pd.Series:
        """Forecast future dividends using Holt-Winters method"""
        try:
            model = ExponentialSmoothing(
                historical_data,
                seasonal_periods=4,
                trend='add',
                seasonal='add'
            )
            fitted_model = model.fit()
            forecast = fitted_model.forecast(periods)
            return forecast
        except Exception as e:
            raise ValueError(f"Error forecasting dividends: {str(e)}")

    def _generate_report(self, 
                        stock_code: str,
                        dividends_df: pd.DataFrame,
                        metrics: Dict,
                        forecast: pd.Series,
                        output_dir: str) -> str:
        """Generate visual report with analysis results"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot historical dividends
            plt.subplot(2, 1, 1)
            dividends_df.plot(kind='bar', title=f'Historical Dividends - {stock_code}')
            plt.xlabel('Date')
            plt.ylabel('Dividend Amount')
            
            # Plot forecast
            plt.subplot(2, 1, 2)
            forecast.plot(kind='bar', color='green', 
                        title='Dividend Forecast (Next Periods)')
            plt.xlabel('Period')
            plt.ylabel('Predicted Dividend')
            
            # Save plot
            filename = f"{output_dir}/{stock_code}_dividend_analysis_{datetime.now().strftime('%Y%m%d')}.png"
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            
            return filename
        except Exception as e:
            raise ValueError(f"Error generating report: {str(e)}")

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute dividend analysis and forecasting
        
        Args:
            stock_code: ASX stock code (e.g., 'CBA.AX')
            start_date: Optional start date for analysis (YYYY-MM-DD)
            forecast_periods: Number of periods to forecast
            output_dir: Directory to save generated reports
            
        Returns:
            Dict[str, Any]: Analysis results including metrics and forecast
        """
        try:
            # Create parameters from kwargs
            parameters = DividendAnalyzerParameters(
                stock_code=kwargs.get("stock_code"),
                start_date=kwargs.get("start_date"),
                forecast_periods=kwargs.get("forecast_periods", 4),
                output_dir=kwargs.get("output_dir", "client_documents/reports")
            )
            
            self._update_progress("Fetching dividend data...", 0.2)
            dividends_df = self._fetch_dividend_data(
                parameters.stock_code, 
                parameters.start_date
            )
            
            self._update_progress("Calculating metrics...", 0.4)
            metrics = self._calculate_metrics(dividends_df)
            
            self._update_progress("Generating forecast...", 0.6)
            forecast = self._forecast_dividends(
                dividends_df.iloc[:, 0], 
                parameters.forecast_periods
            )
            
            self._update_progress("Creating visualization...", 0.8)
            report_path = self._generate_report(
                parameters.stock_code,
                dividends_df,
                metrics,
                forecast,
                parameters.output_dir
            )
            
            self._update_progress("Analysis complete", 1.0)
            
            return {
                "metrics": metrics,
                "forecast": forecast.to_dict(),
                "report_path": report_path
            }
            
        except Exception as e:
            raise ValueError(f"Dividend analysis failed: {str(e)}") 