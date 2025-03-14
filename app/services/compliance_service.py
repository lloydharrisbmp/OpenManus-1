"""
Compliance service for Australian financial regulations.

This module provides functionality for tracking and ensuring compliance with
Australian financial regulations such as those from ASIC, ATO, APRA, and others.

Features:
- Asynchronous file I/O with concurrency control
- Partial success/failure tracking
- Enhanced error handling and logging
- Data integrity validation
- Automatic archiving of old data
"""

import os
import json
import uuid
import asyncio
import aiofiles
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field, validator
from loguru import logger

from app.exceptions import ConfigurationError
from app.dependency_manager import DependencyManager

class ComplianceResult(BaseModel):
    """
    Structured result object for compliance operations with partial success tracking.
    """
    success: bool
    value: Any = None
    error: Optional[str] = None
    source: str = "unknown"  # "file", "memory", "operation"
    duration_ms: Optional[float] = None
    partial_success: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)

    def __bool__(self):
        return self.success

class ComplianceRule(BaseModel):
    """
    Compliance rule model.
    
    Represents a specific compliance rule or requirement.
    """
    rule_id: str
    title: str
    description: str
    regulation: str
    authority: str
    category: str
    tags: List[str] = Field(default_factory=list)
    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    applies_to: List[str] = Field(default_factory=list)
    checklist: List[Dict[str, Any]] = Field(default_factory=list)
    resources: List[Dict[str, str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    severity: str = "medium"  # low, medium, high, critical


class ComplianceCheck(BaseModel):
    """
    Compliance check model.
    
    Represents a compliance check or audit performed against specific rules.
    """
    check_id: str
    tenant_id: str
    rule_ids: List[str]
    performed_by: str
    performed_at: datetime = Field(default_factory=datetime.now)
    status: str  # pending, in_progress, completed
    results: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    reference_documents: List[str] = Field(default_factory=list)
    expiry_date: Optional[date] = None


class ComplianceReport(BaseModel):
    """
    Compliance report model.
    
    Represents a compliance report generated based on compliance checks.
    """
    report_id: str
    tenant_id: str
    title: str
    description: Optional[str] = None
    generated_by: str
    generated_at: datetime = Field(default_factory=datetime.now)
    period_start: date
    period_end: date
    check_ids: List[str] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    file_path: Optional[str] = None


class ComplianceService:
    """
    Enhanced service for managing compliance with Australian financial regulations.
    
    Features:
    - Asynchronous file I/O for scalability
    - Concurrency control for parallel operations
    - Partial success/failure tracking
    - Enhanced error handling and logging
    - Automatic archiving of old data
    """
    
    def __init__(self, data_dir: str = "compliance", concurrency_limit: int = 3, cleanup_interval: int = 3600):
        """
        Initialize the compliance service.
        
        Args:
            data_dir: Directory to store compliance data
            concurrency_limit: Maximum concurrent file operations
            cleanup_interval: Interval in seconds between automatic cleanups
        """
        self.data_dir = data_dir
        self.rules_file = os.path.join(data_dir, "rules.json")
        self.checks_dir = os.path.join(data_dir, "checks")
        self.reports_dir = os.path.join(data_dir, "reports")
        self.archive_dir = os.path.join(data_dir, "archive")
        
        # Create directories
        for directory in [data_dir, self.checks_dir, self.reports_dir, self.archive_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Concurrency control
        self._io_sem = asyncio.Semaphore(concurrency_limit)
        self._cleanup_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=concurrency_limit)
        
        # Initialize data
        self.rules: Dict[str, ComplianceRule] = {}
        self.last_cleanup = datetime.now()
        self.cleanup_interval = cleanup_interval
        
        # Load rules
        self._load_rules_sync()  # Keep sync for initialization
        
        if not self.rules:
            self._initialize_default_rules()
            
        logger.info(f"[ComplianceService] Initialized with data_dir={data_dir}, concurrency_limit={concurrency_limit}")

    async def _load_rules_async(self) -> ComplianceResult:
        """Load compliance rules from file asynchronously."""
        if not os.path.exists(self.rules_file):
            return ComplianceResult(success=True, value={}, source="file")

        start_time = datetime.now()
        try:
            async with self._io_sem:
                async with aiofiles.open(self.rules_file, 'r') as f:
                    content = await f.read()
                
            rules_data = json.loads(content)
            loaded_rules = {}
            partial_success = False
            errors = []

            for rule_id, rule_data in rules_data.items():
                try:
                    # Convert dates
                    for date_field in ['effective_date', 'expiry_date']:
                        if rule_data.get(date_field):
                            rule_data[date_field] = datetime.fromisoformat(rule_data[date_field]).date()
                    for dt_field in ['created_at', 'updated_at']:
                        if rule_data.get(dt_field):
                            rule_data[dt_field] = datetime.fromisoformat(rule_data[dt_field])
                    
                    loaded_rules[rule_id] = ComplianceRule(**rule_data)
                except Exception as e:
                    errors.append(f"Error loading rule {rule_id}: {str(e)}")
                    partial_success = True
                    continue

            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            if not loaded_rules and errors:
                return ComplianceResult(
                    success=False,
                    error="\n".join(errors),
                    source="file",
                    duration_ms=duration
                )

            return ComplianceResult(
                success=True,
                value=loaded_rules,
                partial_success=partial_success,
                details={"errors": errors} if errors else {},
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error loading rules: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def _save_rules_async(self) -> ComplianceResult:
        """Save compliance rules to file asynchronously."""
        start_time = datetime.now()
        try:
            rules_data = {}
            for rule_id, rule in self.rules.items():
                rule_dict = rule.dict()
                # Convert dates to strings
                for date_field in ['effective_date', 'expiry_date']:
                    if rule_dict.get(date_field):
                        rule_dict[date_field] = rule_dict[date_field].isoformat()
                for dt_field in ['created_at', 'updated_at']:
                    rule_dict[dt_field] = rule_dict[dt_field].isoformat()
                
                rules_data[rule_id] = rule_dict

            async with self._io_sem:
                async with aiofiles.open(self.rules_file, 'w') as f:
                    await f.write(json.dumps(rules_data, indent=2))

            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value=len(rules_data),
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error saving rules: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def create_compliance_check_async(self, tenant_id: str, rule_ids: List[str], performed_by: str, **kwargs) -> ComplianceResult:
        """Create a new compliance check asynchronously."""
        start_time = datetime.now()
        check_id = str(uuid.uuid4())
        
        try:
            # Validate rule IDs exist
            invalid_rules = [rule_id for rule_id in rule_ids if rule_id not in self.rules]
            if invalid_rules:
                return ComplianceResult(
                    success=False,
                    error=f"Invalid rule IDs: {invalid_rules}",
                    source="validation"
                )

            check = ComplianceCheck(
                check_id=check_id,
                tenant_id=tenant_id,
                rule_ids=rule_ids,
                performed_by=performed_by,
                status=kwargs.get("status", "pending"),
                **kwargs
            )

            check_file = os.path.join(self.checks_dir, f"{check_id}.json")
            async with self._io_sem:
                async with aiofiles.open(check_file, 'w') as f:
                    check_data = check.dict()
                    check_data['performed_at'] = check_data['performed_at'].isoformat()
                    if check_data.get('expiry_date'):
                        check_data['expiry_date'] = check_data['expiry_date'].isoformat()
                    
                    await f.write(json.dumps(check_data, indent=2))

            # Check if cleanup is needed
            await self._maybe_cleanup()

            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value=check_id,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error creating check: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def get_compliance_check_async(self, check_id: str) -> ComplianceResult:
        """Get a compliance check by ID asynchronously."""
        start_time = datetime.now()
        check_file = os.path.join(self.checks_dir, f"{check_id}.json")
        
        if not os.path.exists(check_file):
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=False,
                error="Check not found",
                source="file",
                duration_ms=duration
            )

        try:
            async with self._io_sem:
                async with aiofiles.open(check_file, 'r') as f:
                    content = await f.read()
                    check_data = json.loads(content)

            # Convert dates
            if check_data.get('performed_at'):
                check_data['performed_at'] = datetime.fromisoformat(check_data['performed_at'])
            if check_data.get('expiry_date'):
                check_data['expiry_date'] = datetime.fromisoformat(check_data['expiry_date']).date()

            check = ComplianceCheck(**check_data)
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ComplianceResult(
                success=True,
                value=check,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error loading check {check_id}: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def get_tenant_compliance_checks_async(self, tenant_id: str) -> ComplianceResult:
        """Get all compliance checks for a tenant asynchronously with partial success tracking."""
        start_time = datetime.now()
        checks = []
        errors = []
        
        try:
            filenames = [f for f in os.listdir(self.checks_dir) if f.endswith('.json')]
            tasks = []
            
            for filename in filenames:
                check_id = filename.replace('.json', '')
                tasks.append(self._load_check_if_tenant(check_id, tenant_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ComplianceResult):
                    if result.success:
                        checks.append(result.value)
                    elif result.error:
                        errors.append(result.error)
                elif isinstance(result, Exception):
                    errors.append(str(result))

            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value=checks,
                partial_success=bool(errors),
                details={"errors": errors} if errors else {},
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error loading tenant checks: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def _maybe_cleanup(self) -> None:
        """Check if cleanup is needed and run it if necessary."""
        now = datetime.now()
        if (now - self.last_cleanup).total_seconds() > self.cleanup_interval:
            if self._cleanup_lock.locked():
                return
                
            async with self._cleanup_lock:
                if (now - self.last_cleanup).total_seconds() > self.cleanup_interval:
                    await self.cleanup_old_data()
                    self.last_cleanup = now

    async def cleanup_old_data(self, days_threshold: int = 90) -> ComplianceResult:
        """
        Archive old compliance data.
        
        Args:
            days_threshold: Number of days after which data is considered old
        """
        start_time = datetime.now()
        archived_files = []
        errors = []
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            # Archive old checks
            check_tasks = []
            for filename in os.listdir(self.checks_dir):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(self.checks_dir, filename)
                check_tasks.append(self._archive_if_old(file_path, cutoff_date, "checks"))
            
            # Archive old reports
            report_tasks = []
            for filename in os.listdir(self.reports_dir):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(self.reports_dir, filename)
                report_tasks.append(self._archive_if_old(file_path, cutoff_date, "reports"))
            
            # Wait for all archive operations
            results = await asyncio.gather(*(check_tasks + report_tasks), return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple):
                    success, file_info = result
                    if success:
                        archived_files.append(file_info)
                    else:
                        errors.append(file_info)
                elif isinstance(result, Exception):
                    errors.append(str(result))

            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value={"archived": archived_files, "errors": errors},
                partial_success=bool(errors),
                source="cleanup",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error during cleanup: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="cleanup",
                duration_ms=duration
            )

    async def _archive_if_old(self, file_path: str, cutoff_date: datetime, data_type: str) -> Tuple[bool, str]:
        """Archive a single file if it's older than the cutoff date."""
        try:
            async with self._io_sem:
                async with aiofiles.open(file_path, 'r') as f:
                    data = json.loads(await f.read())
                
            date_field = 'performed_at' if data_type == 'checks' else 'generated_at'
            if datetime.fromisoformat(data[date_field]) < cutoff_date:
                archive_path = os.path.join(
                    self.archive_dir,
                    data_type,
                    datetime.fromisoformat(data[date_field]).strftime('%Y-%m'),
                    os.path.basename(file_path)
                )
                
                os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                os.rename(file_path, archive_path)
                
                return True, f"Archived {file_path} to {archive_path}"
            
            return True, f"Skipped {file_path} (not old enough)"
        except Exception as e:
            return False, f"Error archiving {file_path}: {str(e)}"

    def _load_rules_sync(self):
        """Load compliance rules from file synchronously."""
        if os.path.exists(self.rules_file):
            try:
                with open(self.rules_file, 'r') as f:
                    rules_data = json.load(f)
                
                for rule_id, rule_data in rules_data.items():
                    # Convert date strings to date objects
                    if 'effective_date' in rule_data and rule_data['effective_date']:
                        rule_data['effective_date'] = datetime.fromisoformat(rule_data['effective_date']).date()
                    if 'expiry_date' in rule_data and rule_data['expiry_date']:
                        rule_data['expiry_date'] = datetime.fromisoformat(rule_data['expiry_date']).date()
                    if 'created_at' in rule_data:
                        rule_data['created_at'] = datetime.fromisoformat(rule_data['created_at'])
                    if 'updated_at' in rule_data:
                        rule_data['updated_at'] = datetime.fromisoformat(rule_data['updated_at'])
                    
                    self.rules[rule_id] = ComplianceRule(**rule_data)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading compliance rules: {e}")
    
    def _save_rules(self):
        """Save compliance rules to file."""
        try:
            rules_data = {}
            for rule_id, rule in self.rules.items():
                rule_dict = rule.dict()
                # Convert dates to strings
                if rule_dict['effective_date']:
                    rule_dict['effective_date'] = rule_dict['effective_date'].isoformat()
                if rule_dict['expiry_date']:
                    rule_dict['expiry_date'] = rule_dict['expiry_date'].isoformat()
                rule_dict['created_at'] = rule_dict['created_at'].isoformat()
                rule_dict['updated_at'] = rule_dict['updated_at'].isoformat()
                
                rules_data[rule_id] = rule_dict
            
            with open(self.rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving compliance rules: {e}")
    
    def _initialize_default_rules(self):
        """Initialize default compliance rules for Australian regulations."""
        # ASIC Rules
        self.add_rule(
            title="Financial Services Guide (FSG) Requirements",
            description="A Financial Services Guide must be provided to all clients and meet ASIC requirements.",
            regulation="Corporations Act 2001 s941A-941F",
            authority="ASIC",
            category="Documentation",
            tags=["FSG", "Disclosure", "ASIC"],
            applies_to=["financial_advisor", "licensee"],
            checklist=[
                {"item": "FSG contains all required information", "details": "Provider details, services offered, fees, etc."},
                {"item": "FSG provided before or at time of advice", "details": "Must be provided at the earliest opportunity"},
                {"item": "FSG updated annually", "details": "Must be reviewed and updated at least annually"}
            ],
            resources=[
                {"name": "ASIC FSG Guidelines", "url": "https://asic.gov.au/regulatory-resources/financial-services/giving-financial-product-advice/"}
            ],
            severity="high"
        )
        
        self.add_rule(
            title="Statement of Advice (SOA) Requirements",
            description="A Statement of Advice must be provided for personal advice and meet ASIC requirements.",
            regulation="Corporations Act 2001 s946A-946C",
            authority="ASIC",
            category="Documentation",
            tags=["SOA", "Advice", "ASIC"],
            applies_to=["financial_advisor"],
            checklist=[
                {"item": "SOA contains all required information", "details": "Purpose of advice, basis for advice, fees, etc."},
                {"item": "SOA provided before or when advice is given", "details": "Must be provided at the time of advice"},
                {"item": "SOA retains a complete copy", "details": "Must keep records for 7 years"}
            ],
            resources=[
                {"name": "ASIC SOA Guidelines", "url": "https://asic.gov.au/regulatory-resources/financial-services/giving-financial-product-advice/"}
            ],
            severity="high"
        )
        
        # ATO Rules
        self.add_rule(
            title="SMSF Compliance Requirements",
            description="Self-Managed Super Funds must comply with ATO regulations and reporting requirements.",
            regulation="Superannuation Industry (Supervision) Act 1993",
            authority="ATO",
            category="SMSF",
            tags=["SMSF", "Superannuation", "ATO"],
            applies_to=["financial_advisor", "smsf_advisor"],
            checklist=[
                {"item": "Annual SMSF return lodged", "details": "Must be lodged by due date"},
                {"item": "SMSF audit completed", "details": "Must be audited by approved SMSF auditor"},
                {"item": "Sole purpose test met", "details": "Fund must be maintained solely for retirement benefits"},
                {"item": "Investment strategy documented", "details": "Must have a documented investment strategy"}
            ],
            resources=[
                {"name": "ATO SMSF Guidelines", "url": "https://www.ato.gov.au/Super/Self-managed-super-funds/"}
            ],
            severity="high"
        )
        
        # AFCA Rules
        self.add_rule(
            title="AFCA Membership Requirements",
            description="Financial service providers must maintain AFCA membership.",
            regulation="Corporations Act 2001 s912A",
            authority="AFCA",
            category="Dispute Resolution",
            tags=["AFCA", "EDR", "Complaints"],
            applies_to=["licensee"],
            checklist=[
                {"item": "AFCA membership current", "details": "Must maintain active membership"},
                {"item": "AFCA contact details in FSG", "details": "Must include AFCA details in disclosure documents"},
                {"item": "Complaint procedures documented", "details": "Must have internal dispute resolution procedures"}
            ],
            resources=[
                {"name": "AFCA Membership Guidelines", "url": "https://www.afca.org.au/about-afca/membership"}
            ],
            severity="medium"
        )
        
        # Anti-Money Laundering
        self.add_rule(
            title="AML/CTF Compliance Requirements",
            description="Financial service providers must comply with Anti-Money Laundering and Counter-Terrorism Financing requirements.",
            regulation="Anti-Money Laundering and Counter-Terrorism Financing Act 2006",
            authority="AUSTRAC",
            category="AML/CTF",
            tags=["AML", "CTF", "AUSTRAC"],
            applies_to=["licensee", "financial_advisor"],
            checklist=[
                {"item": "AML/CTF program documented", "details": "Must have a documented program"},
                {"item": "Customer identification procedures", "details": "Must verify client identity"},
                {"item": "Suspicious matter reporting", "details": "Must report suspicious transactions"},
                {"item": "Staff training program", "details": "Must provide regular staff training"}
            ],
            resources=[
                {"name": "AUSTRAC Guidelines", "url": "https://www.austrac.gov.au/business/how-comply-guidance-and-resources"}
            ],
            severity="critical"
        )
        
        # Privacy Act
        self.add_rule(
            title="Privacy Act Compliance",
            description="Financial service providers must comply with the Privacy Act regarding client information.",
            regulation="Privacy Act 1988",
            authority="OAIC",
            category="Privacy",
            tags=["Privacy", "Data", "OAIC"],
            applies_to=["licensee", "financial_advisor"],
            checklist=[
                {"item": "Privacy policy documented", "details": "Must have a documented privacy policy"},
                {"item": "Privacy policy provided to clients", "details": "Must provide policy to clients"},
                {"item": "Data security measures implemented", "details": "Must have security measures to protect data"},
                {"item": "Data breach response plan", "details": "Must have a plan for responding to data breaches"}
            ],
            resources=[
                {"name": "OAIC Privacy Guidelines", "url": "https://www.oaic.gov.au/privacy/guidance-and-advice/"}
            ],
            severity="high"
        )
    
    def add_rule(self, title: str, description: str, regulation: str, authority: str, category: str, **kwargs) -> str:
        """
        Add a new compliance rule.
        
        Args:
            title: Rule title
            description: Rule description
            regulation: Regulation reference
            authority: Regulatory authority
            category: Rule category
            **kwargs: Additional rule properties
            
        Returns:
            Rule ID
        """
        rule_id = str(uuid.uuid4())
        
        rule = ComplianceRule(
            rule_id=rule_id,
            title=title,
            description=description,
            regulation=regulation,
            authority=authority,
            category=category,
            **kwargs
        )
        
        self.rules[rule_id] = rule
        self._save_rules()
        
        return rule_id
    
    def get_rule(self, rule_id: str) -> Optional[ComplianceRule]:
        """
        Get a compliance rule by ID.
        
        Args:
            rule_id: Rule ID
            
        Returns:
            Compliance rule or None if not found
        """
        return self.rules.get(rule_id)
    
    def get_rules_by_category(self, category: str) -> List[ComplianceRule]:
        """
        Get compliance rules by category.
        
        Args:
            category: Rule category
            
        Returns:
            List of compliance rules
        """
        return [rule for rule in self.rules.values() if rule.category == category]
    
    def get_rules_by_authority(self, authority: str) -> List[ComplianceRule]:
        """
        Get compliance rules by authority.
        
        Args:
            authority: Regulatory authority
            
        Returns:
            List of compliance rules
        """
        return [rule for rule in self.rules.values() if rule.authority == authority]
    
    def get_rules_by_tags(self, tags: List[str]) -> List[ComplianceRule]:
        """
        Get compliance rules by tags.
        
        Args:
            tags: List of tags
            
        Returns:
            List of compliance rules
        """
        return [
            rule for rule in self.rules.values() 
            if any(tag in rule.tags for tag in tags)
        ]
    
    def get_active_rules(self) -> List[ComplianceRule]:
        """
        Get all active compliance rules.
        
        Returns:
            List of active compliance rules
        """
        today = date.today()
        return [
            rule for rule in self.rules.values()
            if (rule.effective_date is None or rule.effective_date <= today) and
               (rule.expiry_date is None or rule.expiry_date >= today)
        ]
    
    def create_compliance_check(self, tenant_id: str, rule_ids: List[str], performed_by: str, **kwargs) -> str:
        """
        Create a new compliance check.
        
        Args:
            tenant_id: Tenant ID
            rule_ids: List of rule IDs to check
            performed_by: User ID who performed the check
            **kwargs: Additional check properties
            
        Returns:
            Check ID
        """
        check_id = str(uuid.uuid4())
        
        check = ComplianceCheck(
            check_id=check_id,
            tenant_id=tenant_id,
            rule_ids=rule_ids,
            performed_by=performed_by,
            status="pending" if not kwargs.get("status") else kwargs.get("status"),
            **kwargs
        )
        
        # Save check to file
        try:
            check_file = os.path.join(self.checks_dir, f"{check_id}.json")
            with open(check_file, 'w') as f:
                check_data = check.dict()
                # Convert dates to strings
                check_data['performed_at'] = check_data['performed_at'].isoformat()
                if check_data['expiry_date']:
                    check_data['expiry_date'] = check_data['expiry_date'].isoformat()
                
                json.dump(check_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving compliance check: {e}")
            return None
        
        return check_id
    
    def get_compliance_check(self, check_id: str) -> Optional[ComplianceCheck]:
        """
        Get a compliance check by ID.
        
        Args:
            check_id: Check ID
            
        Returns:
            Compliance check or None if not found
        """
        check_file = os.path.join(self.checks_dir, f"{check_id}.json")
        
        if not os.path.exists(check_file):
            return None
        
        try:
            with open(check_file, 'r') as f:
                check_data = json.load(f)
            
            # Convert date strings to date objects
            if 'performed_at' in check_data:
                check_data['performed_at'] = datetime.fromisoformat(check_data['performed_at'])
            if 'expiry_date' in check_data and check_data['expiry_date']:
                check_data['expiry_date'] = datetime.fromisoformat(check_data['expiry_date']).date()
            
            return ComplianceCheck(**check_data)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading compliance check: {e}")
            return None
    
    def update_compliance_check(self, check_id: str, **kwargs) -> bool:
        """
        Update a compliance check.
        
        Args:
            check_id: Check ID
            **kwargs: Properties to update
            
        Returns:
            True if successful, False otherwise
        """
        check = self.get_compliance_check(check_id)
        if not check:
            return False
        
        # Update check properties
        for key, value in kwargs.items():
            if hasattr(check, key):
                setattr(check, key, value)
        
        # Save check to file
        try:
            check_file = os.path.join(self.checks_dir, f"{check_id}.json")
            with open(check_file, 'w') as f:
                check_data = check.dict()
                # Convert dates to strings
                check_data['performed_at'] = check_data['performed_at'].isoformat()
                if check_data['expiry_date']:
                    check_data['expiry_date'] = check_data['expiry_date'].isoformat()
                
                json.dump(check_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error updating compliance check: {e}")
            return False
        
        return True
    
    def get_tenant_compliance_checks(self, tenant_id: str) -> List[ComplianceCheck]:
        """
        Get all compliance checks for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of compliance checks
        """
        checks = []
        
        for filename in os.listdir(self.checks_dir):
            if not filename.endswith('.json'):
                continue
            
            check_id = filename.replace('.json', '')
            check = self.get_compliance_check(check_id)
            
            if check and check.tenant_id == tenant_id:
                checks.append(check)
        
        return checks
    
    def create_compliance_report(self, tenant_id: str, title: str, period_start: date, period_end: date, generated_by: str, **kwargs) -> str:
        """
        Create a new compliance report.
        
        Args:
            tenant_id: Tenant ID
            title: Report title
            period_start: Start date of reporting period
            period_end: End date of reporting period
            generated_by: User ID who generated the report
            **kwargs: Additional report properties
            
        Returns:
            Report ID
        """
        report_id = str(uuid.uuid4())
        
        report = ComplianceReport(
            report_id=report_id,
            tenant_id=tenant_id,
            title=title,
            period_start=period_start,
            period_end=period_end,
            generated_by=generated_by,
            **kwargs
        )
        
        # Save report to file
        try:
            report_file = os.path.join(self.reports_dir, f"{report_id}.json")
            with open(report_file, 'w') as f:
                report_data = report.dict()
                # Convert dates to strings
                report_data['generated_at'] = report_data['generated_at'].isoformat()
                report_data['period_start'] = report_data['period_start'].isoformat()
                report_data['period_end'] = report_data['period_end'].isoformat()
                
                json.dump(report_data, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving compliance report: {e}")
            return None
        
        return report_id
    
    def get_compliance_report(self, report_id: str) -> Optional[ComplianceReport]:
        """
        Get a compliance report by ID.
        
        Args:
            report_id: Report ID
            
        Returns:
            Compliance report or None if not found
        """
        report_file = os.path.join(self.reports_dir, f"{report_id}.json")
        
        if not os.path.exists(report_file):
            return None
        
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Convert date strings to date objects
            if 'generated_at' in report_data:
                report_data['generated_at'] = datetime.fromisoformat(report_data['generated_at'])
            if 'period_start' in report_data:
                report_data['period_start'] = datetime.fromisoformat(report_data['period_start']).date()
            if 'period_end' in report_data:
                report_data['period_end'] = datetime.fromisoformat(report_data['period_end']).date()
            
            return ComplianceReport(**report_data)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading compliance report: {e}")
            return None
    
    def get_tenant_compliance_reports(self, tenant_id: str) -> List[ComplianceReport]:
        """
        Get all compliance reports for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of compliance reports
        """
        reports = []
        
        for filename in os.listdir(self.reports_dir):
            if not filename.endswith('.json'):
                continue
            
            report_id = filename.replace('.json', '')
            report = self.get_compliance_report(report_id)
            
            if report and report.tenant_id == tenant_id:
                reports.append(report)
        
        return reports
    
    def generate_compliance_summary(self, tenant_id: str) -> Dict[str, Any]:
        """
        Generate a compliance summary for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            Compliance summary
        """
        summary = {
            "tenant_id": tenant_id,
            "generated_at": datetime.now().isoformat(),
            "total_rules": len(self.get_active_rules()),
            "checks": {
                "total": 0,
                "pending": 0,
                "in_progress": 0,
                "completed": 0,
                "compliance_rate": 0
            },
            "reports": {
                "total": 0,
                "latest": None
            },
            "by_category": {},
            "by_authority": {},
            "high_priority_issues": []
        }
        
        # Get tenant checks
        checks = self.get_tenant_compliance_checks(tenant_id)
        summary["checks"]["total"] = len(checks)
        summary["checks"]["pending"] = len([c for c in checks if c.status == "pending"])
        summary["checks"]["in_progress"] = len([c for c in checks if c.status == "in_progress"])
        summary["checks"]["completed"] = len([c for c in checks if c.status == "completed"])
        
        # Calculate compliance rate for completed checks
        completed_checks = [c for c in checks if c.status == "completed"]
        if completed_checks:
            compliant_items = 0
            total_items = 0
            
            for check in completed_checks:
                for rule_id, result in check.results.items():
                    if isinstance(result, dict) and "items" in result:
                        for item in result["items"]:
                            total_items += 1
                            if item.get("compliant", False):
                                compliant_items += 1
            
            summary["checks"]["compliance_rate"] = (compliant_items / total_items * 100) if total_items > 0 else 0
        
        # Get tenant reports
        reports = self.get_tenant_compliance_reports(tenant_id)
        summary["reports"]["total"] = len(reports)
        
        # Get latest report
        if reports:
            latest_report = max(reports, key=lambda r: r.generated_at)
            summary["reports"]["latest"] = {
                "report_id": latest_report.report_id,
                "title": latest_report.title,
                "generated_at": latest_report.generated_at.isoformat(),
                "period_start": latest_report.period_start.isoformat(),
                "period_end": latest_report.period_end.isoformat()
            }
        
        # Collect statistics by category and authority
        active_rules = self.get_active_rules()
        
        for rule in active_rules:
            # By category
            if rule.category not in summary["by_category"]:
                summary["by_category"][rule.category] = {
                    "total": 0,
                    "compliant": 0,
                    "non_compliant": 0,
                    "pending": 0
                }
            
            summary["by_category"][rule.category]["total"] += 1
            
            # By authority
            if rule.authority not in summary["by_authority"]:
                summary["by_authority"][rule.authority] = {
                    "total": 0,
                    "compliant": 0,
                    "non_compliant": 0,
                    "pending": 0
                }
            
            summary["by_authority"][rule.authority]["total"] += 1
            
            # Count compliance status for each rule
            rule_compliant = False
            rule_non_compliant = False
            
            for check in completed_checks:
                if rule.rule_id in check.rule_ids and rule.rule_id in check.results:
                    result = check.results[rule.rule_id]
                    if isinstance(result, dict) and "compliant" in result:
                        if result["compliant"]:
                            rule_compliant = True
                        else:
                            rule_non_compliant = True
                            
                            # Add to high priority issues if severity is high or critical
                            if rule.severity in ["high", "critical"]:
                                summary["high_priority_issues"].append({
                                    "rule_id": rule.rule_id,
                                    "title": rule.title,
                                    "severity": rule.severity,
                                    "authority": rule.authority,
                                    "category": rule.category,
                                    "check_id": check.check_id,
                                    "details": result.get("details", "")
                                })
            
            if rule_compliant:
                summary["by_category"][rule.category]["compliant"] += 1
                summary["by_authority"][rule.authority]["compliant"] += 1
            elif rule_non_compliant:
                summary["by_category"][rule.category]["non_compliant"] += 1
                summary["by_authority"][rule.authority]["non_compliant"] += 1
            else:
                summary["by_category"][rule.category]["pending"] += 1
                summary["by_authority"][rule.authority]["pending"] += 1
        
        return summary

    async def update_compliance_check_async(self, check_id: str, **kwargs) -> ComplianceResult:
        """Update a compliance check asynchronously."""
        start_time = datetime.now()
        
        # Get existing check
        check_result = await self.get_compliance_check_async(check_id)
        if not check_result.success:
            return check_result
        
        check = check_result.value
        
        try:
            # Update check properties
            for key, value in kwargs.items():
                if hasattr(check, key):
                    setattr(check, key, value)
            
            # Save check to file
            check_file = os.path.join(self.checks_dir, f"{check_id}.json")
            async with self._io_sem:
                async with aiofiles.open(check_file, 'w') as f:
                    check_data = check.dict()
                    check_data['performed_at'] = check_data['performed_at'].isoformat()
                    if check_data.get('expiry_date'):
                        check_data['expiry_date'] = check_data['expiry_date'].isoformat()
                    
                    await f.write(json.dumps(check_data, indent=2))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value=check,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error updating check {check_id}: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def create_compliance_report_async(self, tenant_id: str, title: str, period_start: date, 
                                          period_end: date, generated_by: str, **kwargs) -> ComplianceResult:
        """Create a new compliance report asynchronously."""
        start_time = datetime.now()
        report_id = str(uuid.uuid4())
        
        try:
            report = ComplianceReport(
                report_id=report_id,
                tenant_id=tenant_id,
                title=title,
                period_start=period_start,
                period_end=period_end,
                generated_by=generated_by,
                **kwargs
            )
            
            report_file = os.path.join(self.reports_dir, f"{report_id}.json")
            async with self._io_sem:
                async with aiofiles.open(report_file, 'w') as f:
                    report_data = report.dict()
                    report_data['generated_at'] = report_data['generated_at'].isoformat()
                    report_data['period_start'] = report_data['period_start'].isoformat()
                    report_data['period_end'] = report_data['period_end'].isoformat()
                    
                    await f.write(json.dumps(report_data, indent=2))
            
            # Check if cleanup is needed
            await self._maybe_cleanup()
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value=report_id,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error creating report: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def get_compliance_report_async(self, report_id: str) -> ComplianceResult:
        """Get a compliance report by ID asynchronously."""
        start_time = datetime.now()
        report_file = os.path.join(self.reports_dir, f"{report_id}.json")
        
        if not os.path.exists(report_file):
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=False,
                error="Report not found",
                source="file",
                duration_ms=duration
            )
        
        try:
            async with self._io_sem:
                async with aiofiles.open(report_file, 'r') as f:
                    content = await f.read()
                    report_data = json.loads(content)
            
            # Convert dates
            if report_data.get('generated_at'):
                report_data['generated_at'] = datetime.fromisoformat(report_data['generated_at'])
            if report_data.get('period_start'):
                report_data['period_start'] = datetime.fromisoformat(report_data['period_start']).date()
            if report_data.get('period_end'):
                report_data['period_end'] = datetime.fromisoformat(report_data['period_end']).date()
            
            report = ComplianceReport(**report_data)
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return ComplianceResult(
                success=True,
                value=report,
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error loading report {report_id}: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def get_tenant_compliance_reports_async(self, tenant_id: str) -> ComplianceResult:
        """Get all compliance reports for a tenant asynchronously with partial success tracking."""
        start_time = datetime.now()
        reports = []
        errors = []
        
        try:
            filenames = [f for f in os.listdir(self.reports_dir) if f.endswith('.json')]
            tasks = []
            
            for filename in filenames:
                report_id = filename.replace('.json', '')
                tasks.append(self._load_report_if_tenant(report_id, tenant_id))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ComplianceResult):
                    if result.success:
                        reports.append(result.value)
                    elif result.error:
                        errors.append(result.error)
                elif isinstance(result, Exception):
                    errors.append(str(result))
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value=reports,
                partial_success=bool(errors),
                details={"errors": errors} if errors else {},
                source="file",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error loading tenant reports: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="file",
                duration_ms=duration
            )

    async def generate_compliance_summary_async(self, tenant_id: str) -> ComplianceResult:
        """Generate a compliance summary for a tenant asynchronously."""
        start_time = datetime.now()
        
        try:
            # Get checks and reports concurrently
            checks_result, reports_result = await asyncio.gather(
                self.get_tenant_compliance_checks_async(tenant_id),
                self.get_tenant_compliance_reports_async(tenant_id)
            )
            
            checks = checks_result.value if checks_result.success else []
            reports = reports_result.value if reports_result.success else []
            
            summary = {
                "tenant_id": tenant_id,
                "generated_at": datetime.now().isoformat(),
                "total_rules": len(self.get_active_rules()),
                "checks": {
                    "total": len(checks),
                    "pending": len([c for c in checks if c.status == "pending"]),
                    "in_progress": len([c for c in checks if c.status == "in_progress"]),
                    "completed": len([c for c in checks if c.status == "completed"]),
                    "compliance_rate": 0
                },
                "reports": {
                    "total": len(reports),
                    "latest": None
                },
                "by_category": {},
                "by_authority": {},
                "high_priority_issues": []
            }
            
            # Calculate compliance rate
            completed_checks = [c for c in checks if c.status == "completed"]
            if completed_checks:
                compliant_items = 0
                total_items = 0
                
                for check in completed_checks:
                    for rule_id, result in check.results.items():
                        if isinstance(result, dict) and "items" in result:
                            for item in result["items"]:
                                total_items += 1
                                if item.get("compliant", False):
                                    compliant_items += 1
                
                summary["checks"]["compliance_rate"] = (compliant_items / total_items * 100) if total_items > 0 else 0
            
            # Get latest report
            if reports:
                latest_report = max(reports, key=lambda r: r.generated_at)
                summary["reports"]["latest"] = {
                    "report_id": latest_report.report_id,
                    "title": latest_report.title,
                    "generated_at": latest_report.generated_at.isoformat(),
                    "period_start": latest_report.period_start.isoformat(),
                    "period_end": latest_report.period_end.isoformat()
                }
            
            # Collect statistics by category and authority
            active_rules = self.get_active_rules()
            
            for rule in active_rules:
                # Initialize category stats
                if rule.category not in summary["by_category"]:
                    summary["by_category"][rule.category] = {
                        "total": 0,
                        "compliant": 0,
                        "non_compliant": 0,
                        "pending": 0
                    }
                
                # Initialize authority stats
                if rule.authority not in summary["by_authority"]:
                    summary["by_authority"][rule.authority] = {
                        "total": 0,
                        "compliant": 0,
                        "non_compliant": 0,
                        "pending": 0
                    }
                
                summary["by_category"][rule.category]["total"] += 1
                summary["by_authority"][rule.authority]["total"] += 1
                
                # Check compliance status
                rule_compliant = False
                rule_non_compliant = False
                
                for check in completed_checks:
                    if rule.rule_id in check.rule_ids and rule.rule_id in check.results:
                        result = check.results[rule.rule_id]
                        if isinstance(result, dict) and "compliant" in result:
                            if result["compliant"]:
                                rule_compliant = True
                            else:
                                rule_non_compliant = True
                                
                                # Add to high priority issues
                                if rule.severity in ["high", "critical"]:
                                    summary["high_priority_issues"].append({
                                        "rule_id": rule.rule_id,
                                        "title": rule.title,
                                        "severity": rule.severity,
                                        "authority": rule.authority,
                                        "category": rule.category,
                                        "check_id": check.check_id,
                                        "details": result.get("details", "")
                                    })
                
                # Update statistics
                if rule_compliant:
                    summary["by_category"][rule.category]["compliant"] += 1
                    summary["by_authority"][rule.authority]["compliant"] += 1
                elif rule_non_compliant:
                    summary["by_category"][rule.category]["non_compliant"] += 1
                    summary["by_authority"][rule.authority]["non_compliant"] += 1
                else:
                    summary["by_category"][rule.category]["pending"] += 1
                    summary["by_authority"][rule.authority]["pending"] += 1
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return ComplianceResult(
                success=True,
                value=summary,
                partial_success=not (checks_result.success and reports_result.success),
                details={
                    "check_errors": checks_result.error if not checks_result.success else None,
                    "report_errors": reports_result.error if not reports_result.success else None
                },
                source="summary",
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"[ComplianceService] Error generating summary: {e}", exc_info=True)
            return ComplianceResult(
                success=False,
                error=str(e),
                source="summary",
                duration_ms=duration
            )

    async def _load_check_if_tenant(self, check_id: str, tenant_id: str) -> ComplianceResult:
        """Load a check if it belongs to the specified tenant."""
        result = await self.get_compliance_check_async(check_id)
        if result.success and result.value.tenant_id == tenant_id:
            return result
        return ComplianceResult(success=False, error="Check not found or not owned by tenant", source="file")

    async def _load_report_if_tenant(self, report_id: str, tenant_id: str) -> ComplianceResult:
        """Load a report if it belongs to the specified tenant."""
        result = await self.get_compliance_report_async(report_id)
        if result.success and result.value.tenant_id == tenant_id:
            return result
        return ComplianceResult(success=False, error="Report not found or not owned by tenant", source="file")


# Create global compliance service
compliance_service = ComplianceService()

# Register with dependency manager
DependencyManager.register(ComplianceService, lambda: compliance_service) 