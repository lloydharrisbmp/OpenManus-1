from enum import Enum
from typing import Any, List, Literal, Optional, Union, Dict
from datetime import datetime, date
import re

from pydantic import BaseModel, Field, field_validator, root_validator, ConfigDict

class Role(str, Enum):
    """Message role options"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant" 
    TOOL = "tool"

ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore

class ToolChoice(str, Enum):
    """Tool choice options"""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"

TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore

class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...) # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format for API calls"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None and isinstance(self.tool_calls, list):
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message

    @classmethod
    def user_message(cls, content: str) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(cls, content: Optional[str] = None) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content)

    @classmethod
    def tool_message(cls, content: str, name, tool_call_id: str) -> "Message":
        """Create a tool message"""
        return cls(role=Role.TOOL, content=content, name=name, tool_call_id=tool_call_id)

    @classmethod
    def from_tool_calls(
        cls, tool_calls: List[Any], content: Union[str, List[str]] = "", **kwargs
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT, content=content, tool_calls=formatted_calls, **kwargs
        )


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if self.max_messages is not None and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]


class RiskTolerance(str, Enum):
    """Risk tolerance levels for investment strategies."""
    CONSERVATIVE = "conservative"
    MODERATE_CONSERVATIVE = "moderate_conservative"
    MODERATE = "moderate"
    MODERATE_AGGRESSIVE = "moderate_aggressive"
    AGGRESSIVE = "aggressive"


class EntityType(str, Enum):
    """Types of entities for tax and financial planning."""
    INDIVIDUAL = "individual"
    COMPANY = "company"
    TRUST = "trust"
    SMSF = "smsf"
    PARTNERSHIP = "partnership"


class TaxYear(BaseModel):
    """Australian tax year representation."""
    start_year: int
    end_year: int

    @field_validator('end_year')
    def validate_year_range(cls, v, info):
        if 'start_year' in info.data and v != info.data['start_year'] + 1:
            raise ValueError(f"End year must be exactly one year after start year, got {info.data['start_year']} and {v}")
        return v

    def __str__(self):
        return f"{self.start_year}-{self.end_year}"


class IncomeStream(BaseModel):
    """Income stream with type, amount, and tax details."""
    source: str
    amount: float
    tax_withheld: Optional[float] = 0
    franking_credits: Optional[float] = 0
    tax_deductible: bool = False
    description: Optional[str] = None

    @field_validator('amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v


class FinancialGoal(BaseModel):
    """Financial goal with timeline and details."""
    goal_type: str
    description: str
    target_amount: Optional[float] = None
    timeline_years: Optional[int] = None
    priority: int = Field(ge=1, le=10)
    current_progress: Optional[float] = None


class InvestmentHolding(BaseModel):
    """Investment holding with details about a specific asset."""
    asset_code: str  # Stock ticker, property ID, etc.
    asset_type: str  # stock, bond, property, etc.
    quantity: float
    purchase_price: float
    purchase_date: Optional[date] = None
    current_value: Optional[float] = None
    currency: str = "AUD"
    
    @property
    def gain_loss(self) -> Optional[float]:
        """Calculate gain/loss if current value is known."""
        if self.current_value is not None:
            return self.current_value - (self.purchase_price * self.quantity)
        return None
    
    @property
    def gain_loss_percent(self) -> Optional[float]:
        """Calculate percentage gain/loss if current value is known."""
        if self.current_value is not None and self.purchase_price > 0:
            return ((self.current_value / (self.purchase_price * self.quantity)) - 1) * 100
        return None


class Portfolio(BaseModel):
    """Investment portfolio with holdings and metadata."""
    portfolio_id: str
    name: str
    entity_type: EntityType
    risk_profile: RiskTolerance
    holdings: List[InvestmentHolding] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value based on current values."""
        return sum(h.current_value or 0 for h in self.holdings)
    
    @property
    def allocation_by_asset_type(self) -> Dict[str, float]:
        """Calculate asset allocation by asset type."""
        total = self.total_value
        if total == 0:
            return {}
            
        allocation = {}
        for holding in self.holdings:
            asset_type = holding.asset_type
            value = holding.current_value or 0
            allocation[asset_type] = allocation.get(asset_type, 0) + (value / total * 100)
        
        return allocation


class TaxOptimizationRequest(BaseModel):
    """Request for tax optimization analysis."""
    entity_type: EntityType
    income_streams: Dict[str, float]
    tax_year: Optional[str] = None
    has_spouse: Optional[bool] = False
    has_dependents: Optional[int] = 0
    superannuation_balance: Optional[float] = None
    charitable_donations: Optional[float] = None


class TaxOptimizationResponse(BaseModel):
    """Response from tax optimization analysis."""
    tax_summary: Dict[str, Any]
    optimization_strategies: List[Dict[str, Any]]
    recommended_actions: List[str]
    estimated_tax_savings: Optional[float] = None


class PortfolioOptimizationRequest(BaseModel):
    """Request for portfolio optimization."""
    assets: List[str]
    risk_tolerance: RiskTolerance
    investment_horizon: int = Field(ge=1, le=50)
    tax_entity: EntityType
    existing_holdings: Optional[List[InvestmentHolding]] = None
    ethical_constraints: Optional[List[str]] = None
    target_income: Optional[float] = None


class PortfolioOptimizationResponse(BaseModel):
    """Response from portfolio optimization."""
    optimal_portfolio: Dict[str, float]
    expected_return: float
    expected_risk: float
    asset_allocation: Dict[str, float]
    visualization_path: Optional[str] = None
    sharpe_ratio: Optional[float] = None
    income_estimate: Optional[float] = None


class MarketAnalysisRequest(BaseModel):
    """Request for market analysis."""
    symbols: List[str]
    period: str = "1y"
    metrics: Optional[List[str]] = None


class MarketAnalysisResponse(BaseModel):
    """Response from market analysis."""
    results: Dict[str, Dict[str, Any]]
    analysis_date: datetime = Field(default_factory=datetime.now)


class ReportGenerationRequest(BaseModel):
    """Request for generating a financial report."""
    report_type: str
    title: str
    content: str
    format: str = "markdown"
    include_date: bool = True
    include_disclaimer: bool = True


class ReportGenerationResponse(BaseModel):
    """Response from report generation."""
    file_path: str
    success: bool
    format: str
    url: Optional[str] = None


class ClientProfile(BaseModel):
    """Client profile with personal and financial information."""
    client_id: str
    name: str
    date_of_birth: Optional[date] = None
    email: Optional[str] = None
    risk_profile: RiskTolerance = RiskTolerance.MODERATE
    annual_income: Optional[float] = None
    tax_file_number: Optional[str] = None
    has_spouse: bool = False
    number_of_dependents: int = 0
    goals: List[FinancialGoal] = Field(default_factory=list)
    portfolios: List[Portfolio] = Field(default_factory=list)
    
    @field_validator('tax_file_number')
    def validate_tfn(cls, v):
        if v is not None:
            # Remove spaces for validation
            v_clean = v.replace(" ", "")
            if not re.match(r'^\d{8,9}$', v_clean):
                raise ValueError("Tax File Number must be 8 or 9 digits")
        return v


class WebsiteGenerationRequest(BaseModel):
    """Request for generating a client website."""
    client_name: str
    sections: List[str]
    theme: str = "professional"
    logo_path: Optional[str] = None
    contact_info: Optional[Dict[str, str]] = None


class WebsiteGenerationResponse(BaseModel):
    """Response from website generation."""
    website_path: str
    index_url: str
    success: bool
