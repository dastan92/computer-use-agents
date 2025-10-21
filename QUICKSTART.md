# Quick Start Guide

Get started with Computer Use Agents in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai` - For GPT-5 mini vision API
- `pyautogui` - For screenshot capture and computer control
- `pillow` - For image processing
- `python-dotenv` - For environment variable management

## Step 2: Set Up API Key

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

## Step 3: Run Your First Agent

### Option A: Interactive Mode (Recommended for beginners)

```bash
python agent.py
```

Try these commands:
1. `observe` - See what the agent sees
2. `goal open a browser` - Get action suggestions
3. `click 100 200` - Click at specific coordinates
4. `type hello world` - Type text
5. `quit` - Exit

### Option B: Run Examples

```bash
python example_usage.py
```

Choose from:
1. Simple observation - Just take a screenshot and analyze it
2. Action sequence - Execute multiple actions with screenshots
3. Goal-based - Get AI suggestions for achieving goals
4. Interactive mode - Full control

## How It Works

```
┌─────────────┐
│  Screenshot │ ──> Captured and encoded to base64
└─────────────┘
       │
       v
┌─────────────┐
│ GPT-5 mini  │ ──> Analyzes image and understands content
└─────────────┘
       │
       v
┌─────────────┐
│   Action    │ ──> Suggests or executes keyboard/mouse action
└─────────────┘
       │
       v
┌─────────────┐
│  Screenshot │ ──> Captures result after action
└─────────────┘
```

## Example: Programmatic Usage

```python
from agent import ComputerUseAgent
import os

# Load API key
api_key = os.getenv("OPENAI_API_KEY")

# Create agent
agent = ComputerUseAgent(api_key=api_key, save_screenshots=True)

# Observe current screen
screenshot, base64_img, analysis = agent.observe()
print(f"Analysis: {analysis}")

# Execute an action (screenshot taken automatically after)
agent.execute_action("click", x=500, y=300)

# Get AI suggestion for next step
suggestion = agent.observe_and_act("Open Chrome browser")
print(f"Suggestion: {suggestion}")
```

## Tips

1. **Failsafe**: Move your mouse to the top-left corner to abort any action
2. **Screenshots**: All screenshots are saved to `screenshots/` folder
3. **Coordinates**: Use your mouse position tool to find exact coordinates
4. **Goals**: Be specific in your goals for better action suggestions
5. **Rate Limits**: GPT-5 mini has rate limits - check your OpenAI tier

## Common Use Cases

### 1. Automated Testing
```python
agent.observe_and_act("Click the login button")
agent.execute_action("type", text="user@example.com")
agent.execute_action("press_key", key="tab")
agent.execute_action("type", text="password123")
```

### 2. Screen Monitoring
```python
while True:
    _, _, analysis = agent.observe()
    if "error" in analysis.lower():
        print("Error detected on screen!")
        break
    time.sleep(60)
```

### 3. Guided Navigation
```python
agent.observe_and_act("Navigate to settings")
# Review suggestion and execute
agent.execute_action("click", x=800, y=100)
```

## Troubleshooting

**Problem**: "OPENAI_API_KEY not found"
- **Solution**: Make sure `.env` file exists and contains your API key

**Problem**: "Permission denied" when taking screenshot
- **Solution**: Grant screen recording permissions to your terminal/Python

**Problem**: Mouse/keyboard not working
- **Solution**: Grant accessibility permissions to your terminal/Python

**Problem**: API rate limit exceeded
- **Solution**: Wait a bit or upgrade your OpenAI tier

## Next Steps

- Customize prompts in `vision_analyzer.py`
- Add more action types in `computer_control.py`
- Build automated workflows
- Integrate with other APIs
- Create task-specific agents

## Safety Notes

⚠️ **Important Safety Features:**
- PyAutoGUI failsafe enabled by default
- 0.5 second pause between actions
- All actions logged to console
- Screenshots saved for audit trail

## Resources

- [OpenAI GPT-5 mini docs](https://platform.openai.com/docs/models/gpt-5-mini)
- [PyAutoGUI documentation](https://pyautogui.readthedocs.io/)
- [Full README](README.md)

Happy automating! 🚀

