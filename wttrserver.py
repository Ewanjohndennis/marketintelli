import requests
from fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str):
    """
    Get current weather for a city
    """

    url = f"https://wttr.in/{city}?format=j1"

    response = requests.get(url)
    data = response.json()

    current = data["current_condition"][0]

    return {
        "city": city,
        "temperature_C": current["temp_C"],
        "humidity": current["humidity"],
        "weather": current["weatherDesc"][0]["value"]
    }


if __name__ == "__main__":
    mcp.run()