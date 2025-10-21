"""Computer control module for keyboard and mouse actions."""
import pyautogui
import time


class ComputerControl:
    """Handles keyboard and mouse control actions."""
    
    def __init__(self, failsafe=True):
        """Initialize computer control.
        
        Args:
            failsafe: Enable PyAutoGUI failsafe (move mouse to corner to abort)
        """
        pyautogui.FAILSAFE = failsafe
        pyautogui.PAUSE = 0.5  # Pause between actions
        
        # Get screen size
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen size: {self.screen_width}x{self.screen_height}")
    
    def click(self, x=None, y=None, button='left', clicks=1, interval=0.0):
        """Click the mouse at specified position.
        
        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button ('left', 'right', 'middle')
            clicks: Number of clicks
            interval: Interval between clicks
        """
        if x is not None and y is not None:
            print(f"Clicking at ({x}, {y})")
            pyautogui.click(x, y, clicks=clicks, interval=interval, button=button)
        else:
            print(f"Clicking at current position")
            pyautogui.click(clicks=clicks, interval=interval, button=button)
    
    def double_click(self, x=None, y=None):
        """Double click at specified position."""
        self.click(x, y, clicks=2, interval=0.1)
    
    def right_click(self, x=None, y=None):
        """Right click at specified position."""
        self.click(x, y, button='right')
    
    def type_text(self, text, interval=0.05):
        """Type text using keyboard.
        
        Args:
            text: Text to type
            interval: Interval between keystrokes
        """
        print(f"Typing: {text}")
        pyautogui.write(text, interval=interval)
    
    def press_key(self, key):
        """Press a single key.
        
        Args:
            key: Key name (e.g., 'enter', 'esc', 'tab', 'space')
        """
        print(f"Pressing key: {key}")
        pyautogui.press(key)
    
    def hotkey(self, *keys):
        """Press a combination of keys.
        
        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'c')
        """
        print(f"Pressing hotkey: {'+'.join(keys)}")
        pyautogui.hotkey(*keys)
    
    def move_mouse(self, x, y, duration=0.5):
        """Move mouse to specified position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Time to take for movement
        """
        print(f"Moving mouse to ({x}, {y})")
        pyautogui.moveTo(x, y, duration=duration)
    
    def scroll(self, clicks, x=None, y=None):
        """Scroll the mouse wheel.
        
        Args:
            clicks: Number of clicks to scroll (positive=up, negative=down)
            x: X coordinate to scroll at (optional)
            y: Y coordinate to scroll at (optional)
        """
        print(f"Scrolling {clicks} clicks")
        if x is not None and y is not None:
            pyautogui.scroll(clicks, x, y)
        else:
            pyautogui.scroll(clicks)
    
    def get_mouse_position(self):
        """Get current mouse position.
        
        Returns:
            Tuple of (x, y) coordinates
        """
        return pyautogui.position()
    
    def wait(self, seconds):
        """Wait for specified seconds.
        
        Args:
            seconds: Number of seconds to wait
        """
        print(f"Waiting {seconds} seconds...")
        time.sleep(seconds)

