"""Keyless weather lookups via Open-Meteo (https://open-meteo.com).

Used by the ``get_weather`` tool when no OpenWeatherMap key is configured (or
``[tools.weather].provider = "open-meteo"``), so weather works out of the box.
Results use the same shape as the OpenWeatherMap path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 15.0

# WMO weather interpretation codes (Open-Meteo docs).
_WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class OpenMeteoError(RuntimeError):
    """Lookup failed (unknown place, HTTP error, malformed response)."""


def _get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        response = httpx.get(url, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as exc:
        raise OpenMeteoError(
            f"Open-Meteo request failed: {type(exc).__name__}"
        ) from exc
    except ValueError as exc:
        raise OpenMeteoError("Open-Meteo returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise OpenMeteoError("Open-Meteo returned an unexpected response")
    return data


def geocode(location: str, language: str = "en") -> Dict[str, Any]:
    """Resolve ``"City"`` or ``"City,CC"`` to the best-matching place."""
    name, _, country = (part.strip() for part in location.partition(","))
    params: Dict[str, Any] = {
        "name": name,
        "count": 10 if country else 1,
        "language": language.split("_")[0],
        "format": "json",
    }
    results = _get(_GEOCODE_URL, params).get("results") or []
    if country:
        wanted = country.upper()
        results = [
            r for r in results if str(r.get("country_code", "")).upper() == wanted
        ]
    if not results:
        raise OpenMeteoError(f"Unknown location: {location}")
    return results[0]


def _describe(code: Any) -> str:
    return _WMO_CODES.get(code, "") if isinstance(code, int) else ""


def fetch_weather(
    location: str,
    *,
    units: str = "metric",
    language: str = "en",
    include_forecast: bool = False,
    forecast_hours: int = 24,
) -> Dict[str, Any]:
    """Return current conditions (and optional hourly forecast) for *location*."""
    place = geocode(location, language)
    params: Dict[str, Any] = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "timezone": "auto",
        "current": ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "relative_humidity_2m",
                "weather_code",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
            ]
        ),
        "wind_speed_unit": "mph" if units == "imperial" else "ms",
    }
    if units == "imperial":
        params["temperature_unit"] = "fahrenheit"
        params["precipitation_unit"] = "inch"
    if include_forecast:
        params["hourly"] = ",".join(
            [
                "temperature_2m",
                "apparent_temperature",
                "precipitation_probability",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ]
        )
        params["forecast_hours"] = forecast_hours
    data = _get(_FORECAST_URL, params)
    current = data.get("current") or {}
    result: Dict[str, Any] = {
        "provider": "open-meteo",
        "location": {
            "requested": location,
            "name": place.get("name"),
            "region": place.get("admin1"),
            "country": place.get("country_code"),
            "latitude": place.get("latitude"),
            "longitude": place.get("longitude"),
            "timezone": data.get("timezone"),
        },
        "units": units,
        "current": {
            "time": current.get("time"),
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "description": _describe(current.get("weather_code")),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m"),
            "precipitation": current.get("precipitation"),
        },
    }
    if include_forecast:
        hourly = data.get("hourly") or {}
        times: List[Optional[str]] = hourly.get("time") or []

        def _at(key: str, index: int) -> Any:
            values = hourly.get(key) or []
            return values[index] if index < len(values) else None

        result["forecast"] = [
            {
                "time": stamp,
                "temperature": _at("temperature_2m", i),
                "feels_like": _at("apparent_temperature", i),
                "description": _describe(_at("weather_code", i)),
                "precipitation_probability_percent": _at(
                    "precipitation_probability", i
                ),
                "precipitation": _at("precipitation", i),
                "wind_speed": _at("wind_speed_10m", i),
            }
            for i, stamp in enumerate(times)
        ]
    return result


__all__ = ["OpenMeteoError", "fetch_weather", "geocode"]
