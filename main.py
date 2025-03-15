import asyncio
import os
from pathlib import Path
from datetime import datetime
import re

from app.agent.swe import FinancialPlanningAgent
from app.logger import logger


async def list_conversations(agent):
    """List available conversations."""
    conversations = agent.get_available_conversations()
    
    if not conversations:
        print("\nNo previous conversations found.")
        return None
    
    print("\nAvailable conversations:")
    for i, conv in enumerate(conversations):
        # Format timestamp
        timestamp = conv.get("last_updated", "").split("T")[0]  # Just get the date part
        message_count = conv.get("message_count", 0)
        print(f"{i+1}. [{timestamp}] {conv.get('title')} ({message_count} messages) - ID: {conv.get('id')}")
    
    return conversations


async def load_conversation(agent, conversation_id):
    """Load a specific conversation."""
    try:
        await agent.load_conversation(conversation_id)
        print(f"\nLoaded conversation: {conversation_id}")
        return True
    except Exception as e:
        print(f"\nError loading conversation: {str(e)}")
        return False


async def display_help():
    """Display help information."""
    print("\nAvailable commands:")
    print("  /list - List all available conversations")
    print("  /load [number] - Load a conversation by list number")
    print("  /title [new_title] - Set a title for the current conversation")
    print("  /help - Display this help message")
    print("  /exit or /quit - End the conversation and exit")
    print("\nAny other input will be processed as a request to the agent.")


async def main():
    # Create the agent with disclaimers disabled since the user is a financial adviser
    conversation_title = "Financial Planning Session"
    agent = FinancialPlanningAgent(
        include_disclaimers=False,
        conversation_title=conversation_title
    )
    
    # Create header with version and timestamp
    version = "1.0.0"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    header = f"""
╔══════════════════════════════════════════════════════╗
║ OpenManus Financial Adviser Assistant v{version:<10} ║
║ {timestamp:<48} ║
╚══════════════════════════════════════════════════════╝
"""
    
    try:
        print(header)
        print("Welcome to the Financial Planning Agent. Type '/help' for commands or '/exit' to end.")
        print("Disclaimers are disabled as you are a financial adviser.")
        
        current_conversation = agent.current_conversation_id if hasattr(agent, 'current_conversation_id') else None
        if current_conversation:
            print(f"\nCurrent conversation: {conversation_title} (ID: {current_conversation})")
        
        while True:
            prompt = input("\nEnter your prompt: ")
            
            if not prompt.strip():
                logger.warning("Empty prompt provided.")
                continue
            
            # Check for special commands
            if prompt.startswith("/"):
                command = prompt.lower().split()
                
                if command[0] in ['/exit', '/quit']:
                    logger.info("Ending conversation. Goodbye!")
                    break
                
                elif command[0] == '/list':
                    await list_conversations(agent)
                    continue
                
                elif command[0] == '/load' and len(command) > 1:
                    try:
                        # Check if user provided an index or an ID
                        if command[1].isdigit():
                            # Get conversation by index
                            index = int(command[1]) - 1
                            conversations = agent.get_available_conversations()
                            if 0 <= index < len(conversations):
                                conversation_id = conversations[index]["id"]
                                await load_conversation(agent, conversation_id)
                            else:
                                print(f"Invalid conversation number. Use /list to see available conversations.")
                        else:
                            # Assume it's a conversation ID
                            await load_conversation(agent, command[1])
                    except Exception as e:
                        print(f"Error loading conversation: {str(e)}")
                    continue
                
                elif command[0] == '/title' and len(command) > 1:
                    # Set title for current conversation
                    new_title = " ".join(command[1:])
                    try:
                        if agent.conversation_manager and agent.current_conversation_id:
                            agent.conversation_manager.rename_conversation(agent.current_conversation_id, new_title)
                            print(f"Conversation title set to: {new_title}")
                        else:
                            print("No active conversation to rename.")
                    except Exception as e:
                        print(f"Error setting conversation title: {str(e)}")
                    continue
                
                elif command[0] == '/help':
                    await display_help()
                    continue
                
                else:
                    print(f"Unknown command: {command[0]}. Type /help for available commands.")
                    continue
            
            # Process normal prompt
            logger.warning("Processing your request...")
            response = await agent.process_message(prompt)
            print("\n" + response)
            
            # Check for any visualization files that were generated
            if agent.visualization_paths:
                # Get the files in the current conversation directory
                try:
                    if agent.conversation_manager:
                        files = agent.conversation_manager.get_conversation_files()
                        if files:
                            print("\nFiles generated in this conversation:")
                            for i, file in enumerate(files):
                                print(f"  {i+1}. {file.name}")
                except Exception as e:
                    logger.error(f"Error listing conversation files: {str(e)}")
            
            logger.info("Request processing completed.")
    
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    asyncio.run(main())
