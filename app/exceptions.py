"""
Exception classes for the application.
"""

class BaseFinancialPlannerException(Exception):
    """Base exception for all application-specific exceptions."""
    def __init__(self, message: str = None):
        self.message = message or "An error occurred in the financial planning application"
        super().__init__(self.message)


class ConfigurationError(BaseFinancialPlannerException):
    """Raised when there is an error in the application configuration."""
    def __init__(self, message: str = None):
        self.message = message or "Configuration error"
        super().__init__(self.message)


class ToolExecutionError(BaseFinancialPlannerException):
    """Raised when a tool execution fails."""
    def __init__(self, tool_name: str, message: str = None):
        self.tool_name = tool_name
        self.message = message or f"Error executing tool: {tool_name}"
        super().__init__(self.message)


class AgentError(BaseFinancialPlannerException):
    """Raised when there's an error in agent processing."""
    def __init__(self, message: str = None):
        self.message = message or "Agent processing error"
        super().__init__(self.message)


class FinancialDataError(BaseFinancialPlannerException):
    """Raised when there's an error fetching or processing financial data."""
    def __init__(self, source: str = None, message: str = None):
        self.source = source
        source_info = f" from {source}" if source else ""
        self.message = message or f"Error retrieving financial data{source_info}"
        super().__init__(self.message)


class DocumentProcessingError(BaseFinancialPlannerException):
    """Raised when there's an error processing documents."""
    def __init__(self, file_path: str = None, message: str = None):
        self.file_path = file_path
        file_info = f" ({file_path})" if file_path else ""
        self.message = message or f"Error processing document{file_info}"
        super().__init__(self.message)


class PortfolioAnalysisError(BaseFinancialPlannerException):
    """Raised when there's an error in portfolio analysis."""
    def __init__(self, message: str = None):
        self.message = message or "Error in portfolio analysis"
        super().__init__(self.message)


class WebInterfaceError(BaseFinancialPlannerException):
    """Raised when there's an error in the web interface."""
    def __init__(self, message: str = None):
        self.message = message or "Web interface error"
        super().__init__(self.message)


class SecurityError(BaseFinancialPlannerException):
    """Raised for security-related errors."""
    def __init__(self, message: str = None):
        self.message = message or "Security error"
        super().__init__(self.message)
