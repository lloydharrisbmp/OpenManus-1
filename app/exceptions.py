"""
Exception classes for the application with enhanced error tracking and handling.
"""
from typing import Optional, Dict, Any
from datetime import datetime
import traceback
from enum import Enum
from pydantic import BaseModel

class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorContext(BaseModel):
    timestamp: datetime = datetime.now()
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    source: Optional[str] = None
    trace_id: Optional[str] = None
    additional_info: Dict[str, Any] = {}
    stack_trace: Optional[str] = None

class BaseFinancialPlannerException(Exception):
    """Base exception for all application-specific exceptions."""
    def __init__(
        self,
        message: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        source: Optional[str] = None,
        additional_info: Dict[str, Any] = None
    ):
        self.message = message or "An error occurred in the financial planning application"
        self.context = ErrorContext(
            severity=severity,
            source=source,
            additional_info=additional_info or {},
            stack_trace=traceback.format_exc()
        )
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging and API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "timestamp": self.context.timestamp.isoformat(),
            "severity": self.context.severity,
            "source": self.context.source,
            "additional_info": self.context.additional_info,
            "stack_trace": self.context.stack_trace
        }

class ConfigurationError(BaseFinancialPlannerException):
    """Raised when there is an error in the application configuration."""
    def __init__(
        self,
        message: str = None,
        source: Optional[str] = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "Configuration error",
            severity=ErrorSeverity.HIGH,
            source=source,
            additional_info=additional_info
        )

class ToolExecutionError(BaseFinancialPlannerException):
    """Raised when a tool execution fails."""
    def __init__(
        self,
        tool_name: str,
        message: str = None,
        source: Optional[str] = None,
        additional_info: Dict[str, Any] = None
    ):
        self.tool_name = tool_name
        super().__init__(
            message or f"Error executing tool: {tool_name}",
            severity=ErrorSeverity.MEDIUM,
            source=source,
            additional_info={
                "tool_name": tool_name,
                **(additional_info or {})
            }
        )

class AgentError(BaseFinancialPlannerException):
    """Raised when there's an error in agent processing."""
    def __init__(
        self,
        message: str = None,
        source: Optional[str] = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "Agent processing error",
            severity=ErrorSeverity.HIGH,
            source=source,
            additional_info=additional_info
        )

class FinancialDataError(BaseFinancialPlannerException):
    """Raised when there's an error fetching or processing financial data."""
    def __init__(
        self,
        source: str = None,
        message: str = None,
        additional_info: Dict[str, Any] = None
    ):
        self.data_source = source
        source_info = f" from {source}" if source else ""
        super().__init__(
            message or f"Error retrieving financial data{source_info}",
            severity=ErrorSeverity.HIGH,
            source="financial_data",
            additional_info={
                "data_source": source,
                **(additional_info or {})
            }
        )

class DocumentProcessingError(BaseFinancialPlannerException):
    """Raised when there's an error processing documents."""
    def __init__(
        self,
        file_path: str = None,
        message: str = None,
        additional_info: Dict[str, Any] = None
    ):
        self.file_path = file_path
        file_info = f" ({file_path})" if file_path else ""
        super().__init__(
            message or f"Error processing document{file_info}",
            severity=ErrorSeverity.MEDIUM,
            source="document_processor",
            additional_info={
                "file_path": file_path,
                **(additional_info or {})
            }
        )

class PortfolioAnalysisError(BaseFinancialPlannerException):
    """Raised when there's an error in portfolio analysis."""
    def __init__(
        self,
        message: str = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "Error in portfolio analysis",
            severity=ErrorSeverity.HIGH,
            source="portfolio_analyzer",
            additional_info=additional_info
        )

class WebInterfaceError(BaseFinancialPlannerException):
    """Raised when there's an error in the web interface."""
    def __init__(
        self,
        message: str = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "Web interface error",
            severity=ErrorSeverity.MEDIUM,
            source="web_interface",
            additional_info=additional_info
        )

class SecurityError(BaseFinancialPlannerException):
    """Raised for security-related errors."""
    def __init__(
        self,
        message: str = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "Security error",
            severity=ErrorSeverity.CRITICAL,
            source="security",
            additional_info=additional_info
        )

class ValidationError(BaseFinancialPlannerException):
    """Raised when data validation fails."""
    def __init__(
        self,
        message: str = None,
        field: Optional[str] = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "Validation error",
            severity=ErrorSeverity.MEDIUM,
            source="validator",
            additional_info={
                "field": field,
                **(additional_info or {})
            }
        )

class DatabaseError(BaseFinancialPlannerException):
    """Raised when there's a database-related error."""
    def __init__(
        self,
        message: str = None,
        operation: Optional[str] = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "Database error",
            severity=ErrorSeverity.HIGH,
            source="database",
            additional_info={
                "operation": operation,
                **(additional_info or {})
            }
        )

class APIError(BaseFinancialPlannerException):
    """Raised when there's an error in API communication."""
    def __init__(
        self,
        message: str = None,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        additional_info: Dict[str, Any] = None
    ):
        super().__init__(
            message or "API error",
            severity=ErrorSeverity.HIGH,
            source="api",
            additional_info={
                "endpoint": endpoint,
                "status_code": status_code,
                **(additional_info or {})
            }
        )

class ToolError(Exception):
    """Base exception for tool-related errors."""
    pass

class BashError(ToolError):
    """Exception raised for bash-related errors."""
    pass

class SessionError(BashError):
    """Exception raised for session-related errors."""
    pass

class TimeoutError(BashError):
    """Exception raised when a command times out."""
    pass
