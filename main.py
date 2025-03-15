import asyncio

from app.agent.swe import FinancialPlanningAgent
from app.logger import logger


async def main():
    # Create the agent with disclaimers disabled since the user is a financial adviser
    agent = FinancialPlanningAgent(include_disclaimers=False)
    try:
        print("Welcome to the Financial Planning Agent. Type 'exit' or 'quit' to end the conversation.")
        print("Disclaimers are disabled as you are a financial adviser.")
        while True:
            prompt = input("\nEnter your prompt: ")
            if not prompt.strip():
                logger.warning("Empty prompt provided.")
                continue
                
            if prompt.lower() in ['exit', 'quit']:
                logger.info("Ending conversation. Goodbye!")
                break

            logger.warning("Processing your request...")
            response = await agent.process_message(prompt)
            print("\n" + response)
            logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    asyncio.run(main())
