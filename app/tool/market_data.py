from app.tool.base import BaseTool

class MarketDataTool(BaseTool):
    name = "market_data"
    description = "Tool for fetching market data"
    
    async def execute(self, **kwargs):
        # Implement market data fetching logic here
        return {"success": True, "message": "Market data fetching not implemented yet"} 