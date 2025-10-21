"""Element detection and extraction module using vision AI and image recognition."""
import pyautogui
from PIL import Image
import os
import json
from datetime import datetime


class ElementDetector:
    """Detects, extracts, and locates UI elements on screen."""
    
    def __init__(self, vision_analyzer, elements_dir="elements"):
        """Initialize element detector.
        
        Args:
            vision_analyzer: VisionAnalyzer instance for AI analysis
            elements_dir: Directory to store extracted element images
        """
        self.vision_analyzer = vision_analyzer
        self.elements_dir = elements_dir
        self.elements_cache_file = os.path.join(elements_dir, "elements_cache.json")
        
        if not os.path.exists(elements_dir):
            os.makedirs(elements_dir)
        
        # Load or create element cache
        self.elements_cache = self._load_cache()
    
    def _load_cache(self):
        """Load element cache from disk."""
        if os.path.exists(self.elements_cache_file):
            with open(self.elements_cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save element cache to disk."""
        with open(self.elements_cache_file, 'w') as f:
            json.dump(self.elements_cache, f, indent=2)
    
    def extract_element_from_description(self, screenshot, base64_image, element_description):
        """Use AI to locate and extract a UI element.
        
        Args:
            screenshot: PIL Image of the full screen
            base64_image: Base64 encoded screenshot
            element_description: Description of element to find (e.g., "login button")
            
        Returns:
            Tuple of (cropped PIL Image, coordinates dict) or (None, None) if not found
        """
        screen_width, screen_height = screenshot.size
        
        # Ask AI to estimate element location
        prompt = f"""Find the "{element_description}" on this screen.

Screen size: {screen_width}x{screen_height} pixels

Provide the estimated location as percentages from top-left corner (0,0):
- LEFT: percentage from left edge (0-100)
- TOP: percentage from top edge (0-100)  
- WIDTH: percentage of screen width (0-100)
- HEIGHT: percentage of screen height (0-100)

Format your response EXACTLY like this:
ELEMENT: [name of the element]
LEFT: [number]
TOP: [number]
WIDTH: [number]
HEIGHT: [number]
CONFIDENCE: [low/medium/high]

Be as precise as possible."""
        
        response = self.vision_analyzer.analyze_screenshot(base64_image, prompt)
        print(f"\nAI Element Detection Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)
        
        # Parse the response
        coords = self._parse_coordinates(response, screen_width, screen_height)
        
        if coords:
            # Extract the region
            cropped = screenshot.crop((
                coords['left'],
                coords['top'],
                coords['left'] + coords['width'],
                coords['top'] + coords['height']
            ))
            
            return cropped, coords
        
        return None, None
    
    def _parse_coordinates(self, ai_response, screen_width, screen_height):
        """Parse AI response to extract coordinates.
        
        Args:
            ai_response: Text response from AI
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            Dict with pixel coordinates or None
        """
        try:
            lines = ai_response.strip().split('\n')
            coords = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key in ['LEFT', 'TOP', 'WIDTH', 'HEIGHT']:
                        # Extract number from value
                        import re
                        numbers = re.findall(r'\d+\.?\d*', value)
                        if numbers:
                            coords[key.lower()] = float(numbers[0])
            
            if all(k in coords for k in ['left', 'top', 'width', 'height']):
                # Convert percentages to pixels
                return {
                    'left': int(coords['left'] * screen_width / 100),
                    'top': int(coords['top'] * screen_height / 100),
                    'width': int(coords['width'] * screen_width / 100),
                    'height': int(coords['height'] * screen_height / 100)
                }
        
        except Exception as e:
            print(f"Error parsing coordinates: {e}")
        
        return None
    
    def save_element(self, element_name, cropped_image, coords):
        """Save an extracted element to disk.
        
        Args:
            element_name: Name/description of the element
            cropped_image: PIL Image of the element
            coords: Coordinates dict
            
        Returns:
            Path to saved element image
        """
        # Sanitize element name for filename
        safe_name = "".join(c for c in element_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_').lower()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.png"
        filepath = os.path.join(self.elements_dir, filename)
        
        cropped_image.save(filepath)
        
        # Update cache
        self.elements_cache[element_name] = {
            'filename': filename,
            'filepath': filepath,
            'coords': coords,
            'timestamp': timestamp
        }
        self._save_cache()
        
        print(f"✓ Saved element '{element_name}' to {filepath}")
        return filepath
    
    def find_element_on_screen(self, element_name, confidence=0.8):
        """Find a previously saved element on current screen.
        
        Args:
            element_name: Name of the element to find
            confidence: Matching confidence (0.0-1.0)
            
        Returns:
            Tuple of (x, y) coordinates of element center, or None if not found
        """
        if element_name not in self.elements_cache:
            print(f"Element '{element_name}' not in cache")
            return None
        
        element_info = self.elements_cache[element_name]
        filepath = element_info['filepath']
        
        if not os.path.exists(filepath):
            print(f"Element image not found: {filepath}")
            return None
        
        print(f"Searching for '{element_name}' on screen...")
        
        try:
            # Use PyAutoGUI to find the element
            location = pyautogui.locateOnScreen(filepath, confidence=confidence)
            
            if location:
                center = pyautogui.center(location)
                print(f"✓ Found '{element_name}' at ({center.x}, {center.y})")
                return (center.x, center.y)
            else:
                print(f"✗ Element '{element_name}' not found on screen")
                return None
        
        except Exception as e:
            print(f"Error locating element: {e}")
            return None
    
    def learn_and_click(self, screenshot, base64_image, element_description):
        """Learn about an element and click it in one operation.
        
        Args:
            screenshot: PIL Image of screen
            base64_image: Base64 encoded screenshot
            element_description: Description of element to find and click
            
        Returns:
            Tuple of (success: bool, coordinates: tuple or None)
        """
        print(f"\n{'='*60}")
        print(f"Learning and clicking: {element_description}")
        print('='*60)
        
        # First, try to find it if we already know about it
        coords = self.find_element_on_screen(element_description)
        if coords:
            pyautogui.click(coords[0], coords[1])
            return True, coords
        
        # If not found, extract it from the current screen
        print(f"Element not in cache, extracting from screen...")
        cropped, coord_dict = self.extract_element_from_description(
            screenshot, base64_image, element_description
        )
        
        if cropped and coord_dict:
            # Save the element for future use
            self.save_element(element_description, cropped, coord_dict)
            
            # Click at the center of detected region
            click_x = coord_dict['left'] + coord_dict['width'] // 2
            click_y = coord_dict['top'] + coord_dict['height'] // 2
            
            print(f"Clicking at ({click_x}, {click_y})")
            pyautogui.click(click_x, click_y)
            
            return True, (click_x, click_y)
        
        print(f"✗ Could not locate '{element_description}'")
        return False, None
    
    def list_known_elements(self):
        """List all cached elements.
        
        Returns:
            List of element names
        """
        return list(self.elements_cache.keys())
    
    def clear_cache(self):
        """Clear all cached elements."""
        self.elements_cache = {}
        self._save_cache()
        print("Element cache cleared")

