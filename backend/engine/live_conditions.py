"""
Live weather at ports and airports (Open-Meteo, no API key).

Current wind, gusts and weather code from the forecast API, plus significant wave height from the
marine API for ports. The result is turned into an operational severity, used as a live risk and
delay signal on supplier lanes and shown next to each gateway in the UI.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import httpx

FORECAST = "https://api.open-meteo.com/v1/forecast"
MARINE = "https://marine-api.open-meteo.com/v1/marine"
CACHE_TTL = 30 * 60

# WMO weather codes that disrupt cargo handling on their own (thunderstorms, heavy snow, freezing rain).
SEVERE_CODES = {65, 67, 75, 82, 86, 95, 96, 99}

# (label, threat, expected handling delay in hours)
LEVELS = {
    "severe": ("Severe weather", 0.8, 24.0),
    "high": ("Strong winds / heavy seas", 0.5, 12.0),
    "moderate": ("Windy", 0.25, 4.0),
    "calm": ("Normal conditions", 0.0, 0.0),
}


def assess(gust_kmh: Optional[float], wave_m: Optional[float], code: Optional[int]) -> str:
    g, w = gust_kmh or 0.0, wave_m or 0.0
    if g >= 90 or w >= 6 or code in (95, 96, 99):
        return "severe"
    if g >= 62 or w >= 4 or code in SEVERE_CODES:
        return "high"
    if g >= 50 or w >= 3:
        return "moderate"
    return "calm"


class LiveConditions:
    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout
        self._cache: Dict[tuple, tuple] = {}
        self._lock = threading.Lock()

    def at(self, lat: float, lon: float, marine: bool = False) -> Dict[str, Any]:
        key = (round(lat, 2), round(lon, 2), marine)
        with self._lock:
            hit = self._cache.get(key)
        if hit and time.time() - hit[0] < CACHE_TTL:
            return hit[1]
        try:
            with httpx.Client(timeout=self.timeout) as c:
                cur = c.get(FORECAST, params={"latitude": lat, "longitude": lon, "wind_speed_unit": "kmh",
                                              "current": "temperature_2m,wind_speed_10m,wind_gusts_10m,weather_code"}
                            ).raise_for_status().json()["current"]
                wave = None
                if marine:
                    try:
                        wave = c.get(MARINE, params={"latitude": lat, "longitude": lon, "current": "wave_height"}
                                     ).raise_for_status().json()["current"].get("wave_height")
                    except Exception:
                        wave = None  # inland or outside the marine grid
            level = assess(cur.get("wind_gusts_10m"), wave, cur.get("weather_code"))
            label, threat, delay = LEVELS[level]
            out = {"ok": True, "level": level, "label": label, "threat": threat, "delay_h": delay,
                   "temperature_c": cur.get("temperature_2m"), "wind_kmh": cur.get("wind_speed_10m"),
                   "gust_kmh": cur.get("wind_gusts_10m"), "wave_m": wave, "weather_code": cur.get("weather_code"),
                   "observed_at": cur.get("time"), "source": "Open-Meteo"}
        except Exception as e:
            out = {"ok": False, "level": "unknown", "label": "Live weather unavailable", "threat": 0.0,
                   "delay_h": 0.0, "error": str(e)[:120], "source": "Open-Meteo"}
        with self._lock:
            self._cache[key] = (time.time(), out)
        return out

    def many(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """points: [{lat, lon, marine}] -> conditions in the same order (fetched in parallel)."""
        with ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(lambda p: self.at(p["lat"], p["lon"], p.get("marine", False)), points))
