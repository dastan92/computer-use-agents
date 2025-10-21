"""Screenshot capture module for computer use agent."""
import pyautogui
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
import os


class ScreenCapture:
    """Handles screenshot capture and encoding."""
    
    def __init__(self, save_screenshots=True, screenshots_dir="screenshots"):
        """Initialize screenshot capture.
        
        Args:
            save_screenshots: Whether to save screenshots to disk
            screenshots_dir: Directory to save screenshots
        """
        self.save_screenshots = save_screenshots
        self.screenshots_dir = screenshots_dir
        
        if save_screenshots and not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
    
    def take_screenshot(self, filename=None):
        """Take a screenshot and optionally save it.
        
        Args:
            filename: Optional filename for the screenshot
            
        Returns:
            PIL Image object
        """
        screenshot = pyautogui.screenshot()
        
        if self.save_screenshots:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
            
            filepath = os.path.join(self.screenshots_dir, filename)
            screenshot.save(filepath)
            print(f"Screenshot saved to {filepath}")
        
        return screenshot
    
    def encode_image_to_base64(self, image):
        """Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

