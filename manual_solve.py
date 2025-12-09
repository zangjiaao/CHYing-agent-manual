"""
Manual Challenge Solver Script
==============================

Allows manual input of challenge details to run the solver without
connecting to the competition platform API.
"""
import asyncio
import os
import logging
import uuid
from typing import Dict, Any

from langfuse import get_client
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

from chying_agent.core.singleton import get_config_manager
from chying_agent.task_manager import ChallengeTaskManager
from chying_agent.retry_strategy import RetryStrategy
from chying_agent.challenge_solver import solve_single_challenge
from chying_agent.tools.competition_api_tools import get_api_client
from chying_agent.common import log_system_event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_user_input(prompt: str, default: str = None) -> str:
    """Get input from user with optional default"""
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    else:
        while True:
            result = input(f"{prompt}: ").strip()
            if result:
                return result

async def main():
    load_dotenv()
    
    print("=" * 60)
    print("      CHYing Agent - Manual Challenge Mode")
    print("=" * 60)
    
    # 1. Collect Challenge Details
    print("\n[Input Challenge Details]")
    challenge_code = get_user_input("Challenge Code (ID)", f"manual_{uuid.uuid4().hex[:8]}")
    target_ip = get_user_input("Target IP")
    target_port = get_user_input("Target Port", "80")
    difficulty = get_user_input("Difficulty (easy/medium/hard)", "easy")
    points = int(get_user_input("Points", "100"))
    
    challenge = {
        "challenge_code": challenge_code,
        "difficulty": difficulty,
        "points": points,
        "hint_viewed": False,
        "solved": False,
        "target_info": {
            "ip": target_ip,
            "port": [int(p) for p in target_port.split(",")]
        }
    }
    
    # 2. Initialize Components
    print("\n[Initializing System...]")
    
    try:
        config_manager = get_config_manager()
        config = config_manager.config
        
        # Initialize Langfuse
        try:
            langfuse = get_client()
            langfuse_handler = CallbackHandler()
            log_system_event("[✓] Langfuse Initialized")
        except Exception:
            print("⚠️  Langfuse initialization failed, running without it.")
            langfuse_handler = None

        # Initialize Strategies
        retry_strategy = RetryStrategy(config=config)
        task_manager = ChallengeTaskManager(max_retries=1)
        
        # Initialize API Client in Manual Mode
        api_client = get_api_client()
        api_client.enable_manual_mode()
        api_client.set_manual_challenges([challenge])
        
        # 3. Start Solver
        print(f"\n[Starting Solver for {challenge_code} @ {target_ip}:{target_port}]")
        print("-" * 60)
        
        concurrent_semaphore = asyncio.Semaphore(1)
        
        # Get LLM pair (using initial strategy)
        main_llm, advisor_llm, strategy_desc = retry_strategy.get_llm_pair(0)
        
        result = await solve_single_challenge(
            challenge=challenge,
            main_llm=main_llm,
            advisor_llm=advisor_llm,
            config=config,
            langfuse_handler=langfuse_handler,
            task_manager=task_manager,
            concurrent_semaphore=concurrent_semaphore,
            retry_strategy=retry_strategy,
            attempt_history=[],
            strategy_description=strategy_desc
        )
        
        # 4. Show Result
        print("\n" + "=" * 60)
        print("      EXECUTION FINISHED")
        print("=" * 60)
        
        if result.get("success"):
            print(f"\n✅ SUCCESS!")
            print(f"FLAG: {result.get('flag')}")
            print(f"Score: {result.get('score')}")
        else:
            print(f"\n❌ FAILED")
            print(f"Attempts: {result.get('attempts')}")
            if result.get("error"):
                print(f"Error: {result.get('error')}")
                
        print("-" * 60)

    except Exception as e:
        print(f"\n🚨 Critical Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
