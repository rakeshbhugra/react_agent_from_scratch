
import json
import litellm
from dotenv import load_dotenv

load_dotenv()

def sum_numbers(a: float, b: float) -> float:
    """Add two numbers together and return the sum
    
    Parameters
    ----------
    a : float
        The first number to add
    b : float
        The second number to add
    """
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together and return the product
    
    Parameters
    ----------
    a : float
        The first number to multiply
    b : float
        The second number to multiply
    """
    return a * b

def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather for a specific location
    
    Parameters
    ----------
    location : str
        The city and state, e.g. San Francisco, CA
    unit : str {'celsius', 'fahrenheit'}
        The temperature unit to use
    """
    # Mock weather data
    temp = 72 if unit == "fahrenheit" else 22
    return json.dumps({
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": "sunny"
    })

# Use function_to_dict - much cleaner than manual JSON schema!
AVAILABLE_TOOLS = [
    {"type": "function", "function": litellm.utils.function_to_dict(sum_numbers)},
    {"type": "function", "function": litellm.utils.function_to_dict(multiply_numbers)},
    {"type": "function", "function": litellm.utils.function_to_dict(get_current_weather)}
]

# Function mapping for execution
FUNCTION_MAP = {
    "sum_numbers": sum_numbers,
    "multiply_numbers": multiply_numbers,
    "get_current_weather": get_current_weather
}
