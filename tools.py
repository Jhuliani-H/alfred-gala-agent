"""External (non-RAG) tools for Alfred: web search, weather, and Hub stats."""

import hashlib

from huggingface_hub import HfApi
from smolagents import DuckDuckGoSearchTool, Tool


class SafeWebSearchTool(DuckDuckGoSearchTool):
    """DuckDuckGo search that reports failures instead of raising.

    DuckDuckGo occasionally rate-limits or returns nothing. Returning a message
    (rather than crashing the step) lets the agent rephrase or carry on.
    """

    def forward(self, query: str) -> str:
        try:
            return super().forward(query)
        except Exception as e:
            return f"Web search failed ({type(e).__name__}: {e}). Try a shorter query or rely on other tools."


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = (
        "Fetches the current (simulated) weather for a given location. "
        "Useful for planning outdoor gala activities such as fireworks."
    )
    inputs = {
        "location": {
            "type": "string",
            "description": "The city or place to get weather information for.",
        }
    }
    output_type = "string"

    CONDITIONS = [
        ("Clear skies", 18, "light breeze, 5 km/h"),
        ("Rainy", 12, "gusty, 30 km/h"),
        ("Windy", 15, "strong winds, 45 km/h"),
        ("Partly cloudy", 20, "calm, 3 km/h"),
        ("Thunderstorms", 16, "gusty, 35 km/h"),
    ]

    def forward(self, location: str) -> str:
        # Dummy data: pick a condition from a hash of the location so the same
        # place always gets the same "forecast" (handy for demos and tests).
        digest = hashlib.sha256(location.strip().lower().encode()).digest()
        condition, temp_c, wind = self.CONDITIONS[digest[0] % len(self.CONDITIONS)]
        return (
            f"Simulated weather in {location}: {condition}, {temp_c}°C, wind {wind}. "
            "(Dummy data, not a real forecast.)"
        )


class HubStatsTool(Tool):
    name = "hub_stats"
    description = (
        "Fetches the most downloaded model from a specific author or organization "
        "on the Hugging Face Hub, with its download count."
    )
    inputs = {
        "author": {
            "type": "string",
            "description": "The Hugging Face username or organization, e.g. 'Qwen' or 'facebook'.",
        }
    }
    output_type = "string"

    def forward(self, author: str) -> str:
        try:
            # sort="downloads" returns the most downloaded models first.
            models = list(HfApi().list_models(author=author, sort="downloads", limit=1))
        except Exception as e:
            return f"Error fetching models for {author}: {e}"
        if not models:
            return f"No models found for author '{author}'."
        top = models[0]
        return (
            f"The most downloaded model by {author} is {top.id} "
            f"with {top.downloads:,} downloads."
        )
