"""Vision analysis module using OpenAI GPT-5 mini."""
from openai import OpenAI


class VisionAnalyzer:
    """Analyzes screenshots using OpenAI's GPT-5 mini vision model."""
    
    def __init__(self, api_key):
        """Initialize the vision analyzer.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-5-mini"
    
    def analyze_screenshot(self, base64_image, prompt=None):
        """Analyze a screenshot using GPT-5 mini vision model.
        
        Args:
            base64_image: Base64 encoded image string
            prompt: Custom prompt for the analysis (optional)
            
        Returns:
            String containing the model's analysis
        """
        if prompt is None:
            prompt = """Describe what you see on this screen in detail. 
            Include:
            - Main elements and UI components
            - Text content visible
            - Interactive elements (buttons, links, forms)
            - Current state of the application/window
            - Any notable features or areas of interest"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error analyzing screenshot: {str(e)}"
    
    def analyze_and_suggest_action(self, base64_image, goal):
        """Analyze screenshot and suggest next action based on a goal.
        
        Args:
            base64_image: Base64 encoded image string
            goal: The goal or task to accomplish
            
        Returns:
            String containing suggested action
        """
        prompt = f"""Goal: {goal}

Analyze this screenshot and suggest the next action to take.
Provide a specific action in this format:
ACTION: [click/type/scroll/move/wait]
TARGET: [description of where to click or what to type]
REASON: [why this action helps achieve the goal]

Be specific about coordinates or text to type."""
        
        return self.analyze_screenshot(base64_image, prompt)

