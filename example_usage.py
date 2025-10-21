"""Example usage of the Computer Use Agent."""
from agent import ComputerUseAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your API key.")
    exit(1)


def example_1_simple_observation():
    """Example 1: Simple screen observation."""
    print("Example 1: Simple Screen Observation")
    print("="*60)
    
    agent = ComputerUseAgent(api_key=API_KEY)
    
    # Take a screenshot and analyze it
    screenshot, base64_image, analysis = agent.observe()
    
    print("\nDone!")


def example_2_action_sequence():
    """Example 2: Execute a sequence of actions with screenshots after each."""
    print("\nExample 2: Action Sequence with Screenshots")
    print("="*60)
    
    agent = ComputerUseAgent(api_key=API_KEY)
    
    # Observe initial state
    agent.observe()
    
    # Example: Click at a specific location
    agent.execute_action("click", x=500, y=300)
    
    # Wait a bit
    agent.execute_action("wait", seconds=1)
    
    # Type some text
    agent.execute_action("type", text="Hello from Computer Use Agent!")
    
    # Press Enter
    agent.execute_action("press_key", key="enter")
    
    # Observe final state
    agent.observe()
    
    print(f"\nTotal actions performed: {agent.action_count}")


def example_3_goal_based():
    """Example 3: Goal-based action suggestions."""
    print("\nExample 3: Goal-Based Action Suggestions")
    print("="*60)
    
    agent = ComputerUseAgent(api_key=API_KEY)
    
    # Define a goal
    goal = "Open a web browser"
    
    # Get action suggestion
    action_suggestion = agent.observe_and_act(goal)
    
    # You would then parse the suggestion and execute the appropriate action
    print("\nYou can now manually execute the suggested action.")


def example_4_interactive():
    """Example 4: Run in interactive mode."""
    print("\nExample 4: Interactive Mode")
    print("="*60)
    
    agent = ComputerUseAgent(api_key=API_KEY)
    agent.run_interactive()


if __name__ == "__main__":
    print("Computer Use Agent - Example Usage\n")
    print("Select an example to run:")
    print("1. Simple screen observation")
    print("2. Action sequence with screenshots")
    print("3. Goal-based action suggestions")
    print("4. Interactive mode")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        example_1_simple_observation()
    elif choice == "2":
        example_2_action_sequence()
    elif choice == "3":
        example_3_goal_based()
    elif choice == "4":
        example_4_interactive()
    else:
        print("Invalid choice. Running interactive mode by default.")
        example_4_interactive()

