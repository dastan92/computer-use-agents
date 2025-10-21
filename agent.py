"""Main computer use agent that combines vision, control, and screenshots."""
from screenshot_capture import ScreenCapture
from vision_analyzer import VisionAnalyzer
from computer_control import ComputerControl
from element_detector import ElementDetector
import time


class ComputerUseAgent:
    """Agent that can see the screen and control the computer."""
    
    def __init__(self, api_key, save_screenshots=True, use_element_detection=True):
        """Initialize the computer use agent.
        
        Args:
            api_key: OpenAI API key
            save_screenshots: Whether to save screenshots to disk
            use_element_detection: Enable smart element detection
        """
        print("Initializing Computer Use Agent...")
        self.screen_capture = ScreenCapture(save_screenshots=save_screenshots)
        self.vision_analyzer = VisionAnalyzer(api_key=api_key)
        self.computer_control = ComputerControl(failsafe=True)
        self.action_count = 0
        
        # Initialize element detector if enabled
        self.use_element_detection = use_element_detection
        if use_element_detection:
            self.element_detector = ElementDetector(self.vision_analyzer)
            print("✓ Element detection enabled")
        
        print("Agent initialized successfully!")
    
    def observe(self):
        """Take a screenshot and analyze it.
        
        Returns:
            Tuple of (PIL Image, base64 encoded image, analysis text)
        """
        print(f"\n{'='*60}")
        print(f"Action #{self.action_count + 1} - Taking screenshot...")
        
        # Take screenshot
        screenshot = self.screen_capture.take_screenshot()
        
        # Encode to base64
        base64_image = self.screen_capture.encode_image_to_base64(screenshot)
        
        # Analyze with vision model
        print("Analyzing screenshot with GPT-5 mini...")
        analysis = self.vision_analyzer.analyze_screenshot(base64_image)
        
        print("\nScreen Analysis:")
        print("-" * 60)
        print(analysis)
        print("-" * 60)
        
        return screenshot, base64_image, analysis
    
    def observe_and_act(self, goal):
        """Observe the screen and suggest the next action based on a goal.
        
        Args:
            goal: The goal to accomplish
            
        Returns:
            Suggested action text
        """
        print(f"\n{'='*60}")
        print(f"Action #{self.action_count + 1} - Observing for goal: {goal}")
        
        # Take screenshot
        screenshot = self.screen_capture.take_screenshot()
        
        # Encode to base64
        base64_image = self.screen_capture.encode_image_to_base64(screenshot)
        
        # Analyze and get action suggestion
        print("Analyzing and planning next action...")
        action_suggestion = self.vision_analyzer.analyze_and_suggest_action(
            base64_image, goal
        )
        
        print("\nSuggested Action:")
        print("-" * 60)
        print(action_suggestion)
        print("-" * 60)
        
        return action_suggestion
    
    def execute_action(self, action_type, **kwargs):
        """Execute a computer control action.
        
        Args:
            action_type: Type of action ('click', 'type', 'scroll', 'move', 'wait', etc.)
            **kwargs: Action-specific parameters
        """
        print(f"\nExecuting action: {action_type}")
        
        if action_type == "click":
            self.computer_control.click(**kwargs)
        elif action_type == "double_click":
            self.computer_control.double_click(**kwargs)
        elif action_type == "right_click":
            self.computer_control.right_click(**kwargs)
        elif action_type == "type":
            self.computer_control.type_text(kwargs.get('text', ''))
        elif action_type == "press_key":
            self.computer_control.press_key(kwargs.get('key', 'enter'))
        elif action_type == "hotkey":
            self.computer_control.hotkey(*kwargs.get('keys', []))
        elif action_type == "move":
            self.computer_control.move_mouse(**kwargs)
        elif action_type == "scroll":
            self.computer_control.scroll(kwargs.get('clicks', 1))
        elif action_type == "wait":
            self.computer_control.wait(kwargs.get('seconds', 1))
        else:
            print(f"Unknown action type: {action_type}")
        
        self.action_count += 1
        
        # Take screenshot after action
        print("\nTaking post-action screenshot...")
        self.screen_capture.take_screenshot(
            filename=f"action_{self.action_count:03d}_after.png"
        )
    
    def smart_click(self, element_description):
        """Intelligently locate and click an element by description.
        
        This method uses AI vision to find the element, extracts it as a template,
        and uses image recognition to click it. The template is cached for future use.
        
        Args:
            element_description: Natural language description (e.g., "login button", "chrome icon")
            
        Returns:
            Tuple of (success: bool, coordinates: tuple or None)
        """
        if not self.use_element_detection:
            print("Element detection is disabled. Use execute_action with coordinates instead.")
            return False, None
        
        # Take current screenshot
        screenshot = self.screen_capture.take_screenshot()
        base64_image = self.screen_capture.encode_image_to_base64(screenshot)
        
        # Use element detector to find and click
        success, coords = self.element_detector.learn_and_click(
            screenshot, base64_image, element_description
        )
        
        if success:
            self.action_count += 1
            # Take screenshot after action
            print("\nTaking post-action screenshot...")
            self.screen_capture.take_screenshot(
                filename=f"action_{self.action_count:03d}_after.png"
            )
        
        return success, coords
    
    def list_learned_elements(self):
        """List all elements the agent has learned to recognize.
        
        Returns:
            List of element names
        """
        if not self.use_element_detection:
            return []
        return self.element_detector.list_known_elements()
    
    def run_interactive(self):
        """Run the agent in interactive mode."""
        print("\n" + "="*60)
        print("COMPUTER USE AGENT - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  observe - Take screenshot and analyze")
        print("  goal <your goal> - Get action suggestion for a goal")
        print("  smart_click <element> - AI finds and clicks element (e.g., 'smart_click login button')")
        print("  click <x> <y> - Click at coordinates")
        print("  type <text> - Type text")
        print("  scroll <amount> - Scroll (positive=up, negative=down)")
        print("  wait <seconds> - Wait for seconds")
        print("  list_elements - Show all learned elements")
        print("  quit - Exit")
        print("\n" + "="*60)
        
        while True:
            try:
                command = input("\nEnter command: ").strip()
                
                if not command:
                    continue
                
                if command == "quit":
                    print("Exiting agent...")
                    break
                
                elif command == "observe":
                    self.observe()
                
                elif command.startswith("goal "):
                    goal = command[5:].strip()
                    self.observe_and_act(goal)
                
                elif command.startswith("smart_click "):
                    element = command[12:].strip()
                    self.smart_click(element)
                
                elif command == "list_elements":
                    elements = self.list_learned_elements()
                    if elements:
                        print("\nLearned elements:")
                        for elem in elements:
                            print(f"  - {elem}")
                    else:
                        print("No elements learned yet")
                
                elif command.startswith("click "):
                    parts = command.split()
                    if len(parts) >= 3:
                        x, y = int(parts[1]), int(parts[2])
                        self.execute_action("click", x=x, y=y)
                    else:
                        print("Usage: click <x> <y>")
                
                elif command.startswith("type "):
                    text = command[5:]
                    self.execute_action("type", text=text)
                
                elif command.startswith("scroll "):
                    amount = int(command.split()[1])
                    self.execute_action("scroll", clicks=amount)
                
                elif command.startswith("wait "):
                    seconds = float(command.split()[1])
                    self.execute_action("wait", seconds=seconds)
                
                else:
                    print(f"Unknown command: {command}")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")


if __name__ == "__main__":
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
    
    # Create and run agent
    agent = ComputerUseAgent(api_key=API_KEY, save_screenshots=True)
    agent.run_interactive()

