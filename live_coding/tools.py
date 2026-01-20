import json

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
    temp = 72 if unit == "fahrenheit" else 22
    return json.dumps({
        "location": location,
        "temperature": temp,
        "unit": unit,
        "condition": "sunny"
    })