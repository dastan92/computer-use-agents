# Computer Use Agents

A Python-based system that uses OpenAI's GPT-5 mini vision model to see and understand the computer screen, then control the keyboard and mouse to perform actions.

## Features

- 📸 **Screenshot Capture**: Automatically takes screenshots before and after each action
- 👁️ **Vision Analysis**: Uses GPT-5 mini to understand what's on the screen
- 🖱️ **Computer Control**: Control keyboard and mouse programmatically
- 🎯 **Goal-Based Actions**: Suggest next actions based on specified goals
- 💾 **Action History**: Saves all screenshots with timestamps

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```bash
cp .env.example .env
```

Then edit `.env` and add your API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Components

### 1. `screenshot_capture.py`
Handles taking screenshots and encoding them to base64 for the vision model.

### 2. `vision_analyzer.py`
Integrates with OpenAI's GPT-5 mini vision model to analyze screenshots and suggest actions.

### 3. `computer_control.py`
Provides methods to control keyboard and mouse:
- Click (left, right, double)
- Type text
- Press keys and hotkeys
- Move mouse
- Scroll

### 4. `element_detector.py`
**NEW!** Smart element detection that:
- Uses AI vision to locate UI elements by description
- Automatically crops and saves element templates
- Uses PyAutoGUI image recognition for fast, reliable clicking
- Caches elements for reuse

### 5. `agent.py`
Main agent that combines all components. Takes a screenshot after every action.

### 6. `example_usage.py`
Example scripts showing different usage patterns.

### 7. `demo_smart_click.py`
Interactive demo of smart click functionality.

## Usage

### Interactive Mode

Run the agent in interactive mode:
```bash
python agent.py
```

Commands:
- `observe` - Take screenshot and analyze
- `goal <your goal>` - Get action suggestion for a goal
- `smart_click <element>` - **NEW!** AI finds and clicks element (e.g., "login button")
- `click <x> <y>` - Click at coordinates
- `type <text>` - Type text
- `scroll <amount>` - Scroll (positive=up, negative=down)
- `wait <seconds>` - Wait for seconds
- `list_elements` - Show all learned elements
- `quit` - Exit

### Programmatic Usage

```python
from agent import ComputerUseAgent

# Initialize agent
agent = ComputerUseAgent(
    api_key="your-api-key",
    save_screenshots=True
)

# Observe the screen
screenshot, base64_image, analysis = agent.observe()

# Execute actions (screenshot taken after each action)
agent.execute_action("click", x=500, y=300)
agent.execute_action("type", text="Hello!")
agent.execute_action("press_key", key="enter")

# Get goal-based suggestions
action_suggestion = agent.observe_and_act("Open a web browser")

# NEW! Smart click - AI finds and clicks elements by description
success, coords = agent.smart_click("Chrome icon")
success, coords = agent.smart_click("login button")
success, coords = agent.smart_click("submit form button")

# List all learned elements
elements = agent.list_learned_elements()
print(f"Known elements: {elements}")
```

### Example Scripts

```bash
python example_usage.py
```

Choose from:
1. Simple screen observation
2. Action sequence with screenshots
3. Goal-based action suggestions
4. Interactive mode

## How It Works

### Basic Flow
1. **Observe**: Agent takes a screenshot of the current screen
2. **Analyze**: GPT-5 mini vision model analyzes the screenshot
3. **Decide**: Based on the analysis (and optional goal), determine next action
4. **Act**: Execute the action using keyboard/mouse control
5. **Capture**: Take another screenshot after the action
6. **Repeat**: Continue the cycle

### Smart Click Feature (NEW!)

The smart click feature solves the "how does the agent know where to click?" problem:

**First Time Clicking an Element:**
1. You say: `agent.smart_click("login button")`
2. Agent takes a screenshot
3. GPT-5 mini analyzes and estimates location (e.g., "top-right, ~85% from left, ~10% from top")
4. Agent crops that region and saves it as `login_button_TIMESTAMP.png` in `elements/`
5. Agent clicks the estimated location
6. Template is cached for future use

**Next Time (Much Faster!):**
1. You say: `agent.smart_click("login button")`
2. Agent uses PyAutoGUI's image recognition to find the saved template on screen
3. Clicks instantly - no AI call needed!

**Benefits:**
- ✅ Natural language element descriptions
- ✅ No need to manually specify coordinates
- ✅ Elements are cached and reused
- ✅ Works even if window positions change slightly
- ✅ Much faster on subsequent clicks

## Screenshots

All screenshots are saved to the `screenshots/` directory with timestamps and action numbers.

## Safety

- PyAutoGUI failsafe is enabled by default (move mouse to corner to abort)
- 0.5 second pause between actions
- All actions are logged to console

## Requirements

- Python 3.8+
- OpenAI API key with GPT-5 mini access
- macOS, Windows, or Linux

## Notes

- GPT-5 mini supports images up to 400,000 context window
- Screenshots are automatically resized if needed
- The system works best with clear, specific goals

