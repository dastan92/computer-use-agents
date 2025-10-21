"""Demo of smart click functionality - AI-powered element detection."""
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


def demo_smart_click():
    """Demonstrate smart click functionality."""
    print("="*60)
    print("SMART CLICK DEMO")
    print("="*60)
    print("\nThis demo shows how the agent can:")
    print("1. Use AI to find elements by natural language description")
    print("2. Extract and save element templates")
    print("3. Use image recognition to click them")
    print("4. Cache elements for faster future clicks")
    print("\n" + "="*60)
    
    # Create agent with element detection enabled
    agent = ComputerUseAgent(api_key=API_KEY, save_screenshots=True)
    
    print("\n" + "="*60)
    print("HOW IT WORKS:")
    print("="*60)
    print("""
1. First time you click an element:
   - AI analyzes the screen
   - Estimates element location (as percentages)
   - Crops and saves the element as a template
   - Clicks the element
   
2. Next time you click the same element:
   - Uses PyAutoGUI image recognition
   - Finds the saved template on screen
   - Clicks instantly (much faster!)
    """)
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE:")
    print("="*60)
    
    # Example 1: Click a button by description
    print("\n📌 Example 1: Finding and clicking an element")
    print("-" * 60)
    element_to_find = input("Enter element description (e.g., 'Chrome icon', 'login button'): ").strip()
    
    if element_to_find:
        print(f"\nSearching for: {element_to_find}")
        success, coords = agent.smart_click(element_to_find)
        
        if success:
            print(f"✓ Successfully clicked '{element_to_find}' at {coords}")
        else:
            print(f"✗ Could not find '{element_to_find}'")
    
    # Show learned elements
    print("\n" + "="*60)
    print("LEARNED ELEMENTS:")
    print("="*60)
    elements = agent.list_learned_elements()
    if elements:
        print("\nThe agent has learned these elements:")
        for i, elem in enumerate(elements, 1):
            print(f"{i}. {elem}")
        
        print("\n💡 Tip: Next time you use smart_click with these elements,")
        print("   it will be much faster using image recognition!")
    else:
        print("No elements learned yet.")
    
    # Example 2: Try another element
    print("\n" + "="*60)
    print("TRY ANOTHER ELEMENT:")
    print("="*60)
    another = input("\nWant to try finding another element? (y/n): ").strip().lower()
    
    if another == 'y':
        element_to_find = input("Enter element description: ").strip()
        if element_to_find:
            success, coords = agent.smart_click(element_to_find)
            if success:
                print(f"✓ Successfully clicked '{element_to_find}' at {coords}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nCheck these folders:")
    print("  - screenshots/ : All screenshots taken")
    print("  - elements/ : Extracted element templates")
    print("\nYou can now use these elements in your scripts!")


if __name__ == "__main__":
    try:
        demo_smart_click()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")

