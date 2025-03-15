"""
Example workflows combining browser automation with Microsoft services.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
from pathlib import Path

from app.tool.browser_use_tool import BrowserUseTool
from app.tool.ms_collaboration import SharePointListManager, TeamsMessaging
from app.tool.dataverse_query import DataverseQueryBuilder

logger = logging.getLogger(__name__)

class AutomatedWorkflows:
    """Example workflows combining browser automation with Microsoft services."""
    
    def __init__(
        self,
        browser_tool: BrowserUseTool,
        sharepoint_manager: SharePointListManager,
        teams_messaging: TeamsMessaging
    ):
        self.browser = browser_tool
        self.sharepoint = sharepoint_manager
        self.teams = teams_messaging
    
    async def web_form_automation(
        self,
        form_url: str,
        dataverse_query: DataverseQueryBuilder,
        notification_channel: Optional[str] = None
    ) -> bool:
        """Automate web form filling using Dataverse data and notify on Teams."""
        try:
            # Get records from Dataverse
            records = await dataverse_query.execute()
            if not records:
                logger.warning("No records found in Dataverse")
                return False
            
            # Navigate to form
            await self.browser.navigate(form_url)
            
            # Process each record
            for record in records:
                # Fill form fields
                field_mappings = {
                    "#name": record.get("name"),
                    "#email": record.get("email"),
                    "#company": record.get("company"),
                    "#message": record.get("message")
                }
                
                await self.browser.fill_form(field_mappings)
                
                # Take screenshot before submission
                screenshot = await self.browser.take_screenshot()
                
                # Submit form
                await self.browser.click_element("button[type='submit']")
                
                # Wait for confirmation
                success = await self.browser.wait_for_element(".success-message", timeout=10)
                
                if success and notification_channel:
                    # Create SharePoint list item for tracking
                    await self.sharepoint.create_list_item({
                        "FormURL": form_url,
                        "RecordID": record.get("id"),
                        "SubmissionTime": datetime.now().isoformat(),
                        "Status": "Success"
                    })
                    
                    # Send Teams notification with screenshot
                    await self.teams.send_channel_message(
                        content=f"Form submitted successfully for {record.get('name')}",
                        attachments=[{
                            "contentType": "image/png",
                            "contentUrl": screenshot
                        }]
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Form automation failed: {str(e)}")
            if notification_channel:
                await self.teams.send_channel_message(
                    content=f"❌ Form automation failed: {str(e)}",
                    importance="high"
                )
            return False
    
    async def data_scraping_workflow(
        self,
        target_url: str,
        sharepoint_list_name: str,
        teams_channel: Optional[str] = None
    ) -> bool:
        """Scrape web data, store in SharePoint list, and notify on Teams."""
        try:
            # Navigate to target URL
            await self.browser.navigate(target_url)
            
            # Extract data using configured selectors
            data = await self.browser.extract_data({
                "title": "h1.page-title",
                "price": ".product-price",
                "description": ".product-description",
                "availability": ".stock-status"
            })
            
            # Take screenshot of the page
            screenshot = await self.browser.take_screenshot()
            
            # Create SharePoint list item
            list_item = await self.sharepoint.create_list_item({
                "Title": data.get("title"),
                "Price": data.get("price"),
                "Description": data.get("description"),
                "Availability": data.get("availability"),
                "LastUpdated": datetime.now().isoformat(),
                "SourceURL": target_url
            })
            
            if teams_channel and list_item:
                # Send Teams notification
                await self.teams.send_channel_message(
                    content=f"""
                    ✅ New data scraped and stored:
                    - Title: {data.get('title')}
                    - Price: {data.get('price')}
                    - Availability: {data.get('availability')}
                    """,
                    attachments=[{
                        "contentType": "image/png",
                        "contentUrl": screenshot
                    }]
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Data scraping workflow failed: {str(e)}")
            if teams_channel:
                await self.teams.send_channel_message(
                    content=f"❌ Data scraping failed: {str(e)}",
                    importance="high"
                )
            return False
    
    async def automated_testing_workflow(
        self,
        test_cases: List[Dict[str, Any]],
        teams_channel: Optional[str] = None
    ) -> bool:
        """Run automated tests and report results to Teams."""
        try:
            results = []
            
            for test_case in test_cases:
                # Navigate to test URL
                await self.browser.navigate(test_case["url"])
                
                # Perform test actions
                for action in test_case["actions"]:
                    if action["type"] == "click":
                        await self.browser.click_element(action["selector"])
                    elif action["type"] == "input":
                        await self.browser.fill_form({
                            action["selector"]: action["value"]
                        })
                    elif action["type"] == "wait":
                        await self.browser.wait_for_element(
                            action["selector"],
                            timeout=action.get("timeout", 10)
                        )
                
                # Verify expected conditions
                success = True
                error_message = None
                
                for assertion in test_case["assertions"]:
                    try:
                        if assertion["type"] == "element_present":
                            await self.browser.wait_for_element(
                                assertion["selector"],
                                timeout=assertion.get("timeout", 5)
                            )
                        elif assertion["type"] == "element_text":
                            element_text = await self.browser.get_element_text(
                                assertion["selector"]
                            )
                            assert element_text == assertion["expected_text"]
                    except Exception as e:
                        success = False
                        error_message = str(e)
                        break
                
                # Take screenshot
                screenshot = await self.browser.take_screenshot()
                
                # Store result
                results.append({
                    "test_name": test_case["name"],
                    "success": success,
                    "error": error_message,
                    "screenshot": screenshot,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Create SharePoint list item for test result
                await self.sharepoint.create_list_item({
                    "TestName": test_case["name"],
                    "Success": success,
                    "ErrorMessage": error_message,
                    "ExecutionTime": datetime.now().isoformat(),
                    "URL": test_case["url"]
                })
            
            if teams_channel:
                # Prepare summary message
                total_tests = len(results)
                passed_tests = sum(1 for r in results if r["success"])
                failed_tests = total_tests - passed_tests
                
                summary = f"""
                🧪 Test Execution Summary:
                - Total Tests: {total_tests}
                - Passed: ✅ {passed_tests}
                - Failed: ❌ {failed_tests}
                
                Detailed Results:
                """
                
                for result in results:
                    status = "✅" if result["success"] else "❌"
                    summary += f"\n{status} {result['test_name']}"
                    if not result["success"]:
                        summary += f"\n   Error: {result['error']}"
                
                # Send Teams message with summary and screenshots
                await self.teams.send_channel_message(
                    content=summary,
                    importance="high" if failed_tests > 0 else "normal",
                    attachments=[
                        {
                            "contentType": "image/png",
                            "contentUrl": result["screenshot"]
                        }
                        for result in results if not result["success"]
                    ]
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Testing workflow failed: {str(e)}")
            if teams_channel:
                await self.teams.send_channel_message(
                    content=f"❌ Testing workflow failed: {str(e)}",
                    importance="high"
                )
            return False 