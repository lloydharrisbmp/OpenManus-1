import os
from typing import Dict, List, Optional
import re
from datetime import datetime
import io

from pydantic import Field
import pandas as pd
import PyPDF2
from app.tool.base import BaseTool


class DocumentAnalyzerTool(BaseTool):
    """Tool for analyzing and extracting information from financial documents."""

    name: str = "document_analyzer"
    description: str = "Analyzes and extracts information from financial documents (PDF, CSV, Excel)"
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the document file to analyze",
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform (e.g., 'statement_summary', 'portfolio_holdings', 'transaction_history', 'tax_document')",
            },
            "date_range": {
                "type": "string",
                "description": "Optional date range for filtering data (format: 'YYYY-MM-DD to YYYY-MM-DD')",
            }
        },
        "required": ["file_path"]
    }

    async def execute(self, file_path: str, analysis_type: str = "general", date_range: Optional[str] = None) -> Dict:
        """Execute document analysis based on file type and analysis type."""
        try:
            if not os.path.exists(file_path):
                return {"observation": f"Error: File not found at path {file_path}", "success": False}
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Process based on file type
            if file_ext == '.pdf':
                result = await self._analyze_pdf(file_path, analysis_type)
            elif file_ext == '.csv':
                result = await self._analyze_csv(file_path, analysis_type, date_range)
            elif file_ext in ['.xlsx', '.xls']:
                result = await self._analyze_excel(file_path, analysis_type, date_range)
            else:
                return {"observation": f"Unsupported file type: {file_ext}", "success": False}
            
            # Add general file information
            file_info = {
                "file_name": os.path.basename(file_path),
                "file_size": f"{os.path.getsize(file_path) / 1024:.2f} KB",
                "analysis_type": analysis_type,
                "date_analyzed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if date_range:
                file_info["date_range"] = date_range
                
            result["file_info"] = file_info
            return result
            
        except Exception as e:
            return {"observation": f"Error analyzing document: {str(e)}", "success": False}
    
    async def _analyze_pdf(self, file_path: str, analysis_type: str) -> Dict:
        """Analyze a PDF document."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                # Extract text from all pages
                full_text = ""
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    full_text += page.extract_text() + "\n\n"
                
                # Perform analysis based on type
                if analysis_type == "statement_summary":
                    return await self._analyze_statement(full_text)
                elif analysis_type == "portfolio_holdings":
                    return await self._analyze_portfolio(full_text)
                elif analysis_type == "tax_document":
                    return await self._analyze_tax_document(full_text)
                else:  # General analysis
                    return await self._general_pdf_analysis(full_text, num_pages)
                
        except Exception as e:
            return {"observation": f"Error analyzing PDF: {str(e)}", "success": False}
    
    async def _analyze_csv(self, file_path: str, analysis_type: str, date_range: Optional[str] = None) -> Dict:
        """Analyze a CSV document."""
        try:
            df = pd.read_csv(file_path)
            
            # Filter by date range if provided
            if date_range and any(col.lower() in ["date", "transaction_date", "trade_date"] for col in df.columns):
                date_col = next(col for col in df.columns if col.lower() in ["date", "transaction_date", "trade_date"])
                start_date, end_date = date_range.split(" to ")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            
            # Generate summary statistics
            summary = {
                "row_count": len(df),
                "column_names": list(df.columns),
                "data_preview": df.head(5).to_dict(orient='records'),
            }
            
            if analysis_type == "transaction_history":
                # Add transaction-specific analysis
                if any(col.lower() in ["amount", "value", "transaction_amount"] for col in df.columns):
                    amount_col = next(col for col in df.columns if col.lower() in ["amount", "value", "transaction_amount"])
                    summary["total_value"] = df[amount_col].sum()
                    summary["average_transaction"] = df[amount_col].mean()
                    summary["largest_transaction"] = df[amount_col].max()
            
            return {"observation": str(summary), "success": True}
            
        except Exception as e:
            return {"observation": f"Error analyzing CSV: {str(e)}", "success": False}
    
    async def _analyze_excel(self, file_path: str, analysis_type: str, date_range: Optional[str] = None) -> Dict:
        """Analyze an Excel document."""
        try:
            # Read the Excel file
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Process the first sheet or specific sheet based on analysis type
            sheet_to_analyze = sheet_names[0]  # Default to first sheet
            
            # Read the selected sheet
            df = pd.read_excel(file_path, sheet_name=sheet_to_analyze)
            
            # Filter by date range if provided
            if date_range and any(col.lower() in ["date", "transaction_date", "trade_date"] for col in df.columns):
                date_col = next(col for col in df.columns if col.lower() in ["date", "transaction_date", "trade_date"])
                start_date, end_date = date_range.split(" to ")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            
            # Generate summary of the Excel file
            summary = {
                "sheet_names": sheet_names,
                "analyzed_sheet": sheet_to_analyze,
                "row_count": len(df),
                "column_names": list(df.columns),
                "data_preview": df.head(5).to_dict(orient='records')
            }
            
            # Add specific analysis based on analysis_type
            if analysis_type == "portfolio_holdings":
                if any(col.lower() in ["value", "market_value", "amount"] for col in df.columns):
                    value_col = next(col for col in df.columns if col.lower() in ["value", "market_value", "amount"])
                    summary["total_portfolio_value"] = df[value_col].sum()
                    
                    # Try to get asset allocation if possible
                    if any(col.lower() in ["asset_class", "category", "type"] for col in df.columns):
                        asset_col = next(col for col in df.columns if col.lower() in ["asset_class", "category", "type"])
                        allocation = df.groupby(asset_col)[value_col].sum()
                        summary["asset_allocation"] = allocation.to_dict()
            
            return {"observation": str(summary), "success": True}
            
        except Exception as e:
            return {"observation": f"Error analyzing Excel file: {str(e)}", "success": False}
    
    async def _general_pdf_analysis(self, text: str, num_pages: int) -> Dict:
        """General analysis of PDF content."""
        # Extract basic information
        financial_summary = {
            "page_count": num_pages,
            "word_count": len(text.split()),
            "detected_entities": {}
        }
        
        # Look for Australian financial entities
        aus_entities = ["ASX", "ATO", "ASIC", "SMSF", "superfund", "superannuation", 
                        "franking credit", "dividend", "capital gain", "trust distribution"]
        for entity in aus_entities:
            count = len(re.findall(r'\b' + re.escape(entity) + r'\b', text, re.IGNORECASE))
            if count > 0:
                financial_summary["detected_entities"][entity] = count
        
        # Look for dollar amounts
        dollar_amounts = re.findall(r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
        if dollar_amounts:
            financial_summary["dollar_amounts_detected"] = len(dollar_amounts)
            # Try to find largest values
            try:
                numeric_values = [float(amount.replace(',', '')) for amount in dollar_amounts]
                if numeric_values:
                    financial_summary["largest_dollar_amount"] = max(numeric_values)
                    financial_summary["smallest_dollar_amount"] = min(numeric_values)
            except:
                pass
        
        # Look for dates
        dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
        if dates:
            financial_summary["dates_detected"] = len(dates)
        
        # Extract a sample of the content
        sample_text = text[:500] + "..." if len(text) > 500 else text
        financial_summary["content_sample"] = sample_text
        
        return {"observation": str(financial_summary), "success": True}
    
    async def _analyze_statement(self, text: str) -> Dict:
        """Analyze financial statement content."""
        statement_info = {
            "document_type": "financial_statement",
            "extracted_data": {}
        }
        
        # Look for account information
        account_match = re.search(r'Account\s*(?:Number|No|#)?[:.\s]*\s*([A-Za-z0-9-]+)', text)
        if account_match:
            statement_info["extracted_data"]["account_number"] = account_match.group(1)
        
        # Look for statement period
        period_match = re.search(r'(?:Statement|Period)(?:\s*for)?\s*(?:the\s*period)?\s*(?:from|:)?\s*([A-Za-z0-9\s,]+\s*to\s*[A-Za-z0-9\s,]+)', text)
        if period_match:
            statement_info["extracted_data"]["statement_period"] = period_match.group(1).strip()
        
        # Look for opening and closing balances
        opening_match = re.search(r'Opening\s*Balance[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
        if opening_match:
            statement_info["extracted_data"]["opening_balance"] = opening_match.group(1)
        
        closing_match = re.search(r'Closing\s*Balance[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
        if closing_match:
            statement_info["extracted_data"]["closing_balance"] = closing_match.group(1)
        
        # Look for total deposits/credits
        deposits_match = re.search(r'(?:Total\s*)?(?:Deposits|Credits)[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
        if deposits_match:
            statement_info["extracted_data"]["total_deposits"] = deposits_match.group(1)
        
        # Look for total withdrawals/debits
        withdrawals_match = re.search(r'(?:Total\s*)?(?:Withdrawals|Debits)[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
        if withdrawals_match:
            statement_info["extracted_data"]["total_withdrawals"] = withdrawals_match.group(1)
        
        return {"observation": str(statement_info), "success": True}
    
    async def _analyze_portfolio(self, text: str) -> Dict:
        """Analyze portfolio holdings content."""
        portfolio_info = {
            "document_type": "portfolio_statement",
            "holdings": [],
            "summary": {}
        }
        
        # Look for total portfolio value
        total_value_match = re.search(r'(?:Total|Portfolio|Market)\s*(?:Value|Worth|Holdings)[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', text)
        if total_value_match:
            portfolio_info["summary"]["total_value"] = total_value_match.group(1)
        
        # Look for individual holdings - this is a simplified approach
        # A more robust approach would use table extraction algorithms
        holding_pattern = r'([A-Za-z0-9\s.]+)\s+([A-Z]{1,5})[^\n$]*\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'
        holdings = re.findall(holding_pattern, text)
        
        for holding in holdings[:10]:  # Limit to first 10 matches to avoid false positives
            if len(holding) >= 3:
                portfolio_info["holdings"].append({
                    "name": holding[0].strip(),
                    "ticker": holding[1].strip(),
                    "value": holding[2].strip()
                })
        
        return {"observation": str(portfolio_info), "success": True}
    
    async def _analyze_tax_document(self, text: str) -> Dict:
        """Analyze tax document content."""
        tax_info = {
            "document_type": "tax_document",
            "tax_year": None,
            "extracted_data": {}
        }
        
        # Look for tax year
        tax_year_match = re.search(r'(?:Tax|Income)\s*(?:Year|Period)[:.]*\s*(?:\d{4}[-/]\d{4}|\d{4})', text)
        if tax_year_match:
            tax_info["tax_year"] = tax_year_match.group(0).split(":")[-1].strip()
        
        # Look for common Australian tax items
        tax_items = {
            "gross_income": r'Gross\s*(?:Income|Salary|Earnings)[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            "tax_withheld": r'Tax\s*Withheld[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            "franking_credits": r'Franking\s*Credits?[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            "capital_gains": r'(?:Net\s*)?Capital\s*Gains[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            "deductions": r'(?:Total\s*)?Deductions[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            "taxable_income": r'Taxable\s*Income[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            "medicare_levy": r'Medicare\s*Levy[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
            "tax_offset": r'(?:Tax\s*)?Offset(?:s)?[:.\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
        }
        
        for item_key, pattern in tax_items.items():
            match = re.search(pattern, text)
            if match:
                tax_info["extracted_data"][item_key] = match.group(1)
        
        return {"observation": str(tax_info), "success": True} 