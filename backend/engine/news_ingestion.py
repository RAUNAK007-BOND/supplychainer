import feedparser
import urllib.parse
import time
import socket
from typing import List, Dict

class DynamicNewsIngestor:
    """
    Supplychainer Stage 1: Dynamic News Ingestion.
    Consumes live external intelligence from Google News RSS.
    Implements location-aware text retrieval and caching.
    """
    def __init__(self):
        self.cache = {} # {query: (timestamp, content)}
        self.cache_ttl = 900 # 15 minutes
        
        # Offline baselines, used when no live intelligence is available. These must be neutral:
        # the previous texts asserted congestion/backlogs on every corridor, and once the NLP
        # stage actually worked they scored ~0.9 severity on every sea lane with zero evidence.
        self.fallback_news = {
            "sea": "Vessel schedules and terminal operations are running normally with no reported disruptions.",
            "air": "Air cargo handling operating within normal parameters at international hubs.",
            "road": "Highway freight corridors operating normally.",
            "rail": "Rail freight services running to normal schedules."
        }

    def get_latest_news(self, location: str, transport_mode: str) -> str:
        """
        Fetches live news for a specific geographic node and transport mode.
        """
        return self.get_intel(location, transport_mode)["text"]

    def get_intel(self, location: str, transport_mode: str) -> Dict:
        """Structured variant: {text, headlines, source: LIVE|CACHE|FALLBACK, fetched_at}."""
        query = f"{location} {transport_mode} logistics disruption"
        
        # 1. Check Cache
        now = time.time()
        if query in self.cache:
            ts, content = self.cache[query]
            if now - ts < self.cache_ttl:
                return {"text": content, "headlines": content.split(" | "), "source": "CACHE", "fetched_at": ts}

        # 2. Live Ingestion (Google News RSS)
        t_start = time.perf_counter()
        print(f"[TRACE] STEP 7: News ingestion started for {location}")
        try:
            encoded_query = urllib.parse.quote(query)
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            
            # Set a hard timeout for the socket
            socket.setdefaulttimeout(2.0)
            
            feed = feedparser.parse(rss_url)
            
            if feed.entries:
                top_headlines = [entry.title for entry in feed.entries[:3]]
                content = " | ".join(top_headlines)
                self.cache[query] = (now, content)
                print(f"[TRACE] STEP 8: News ingestion complete ({time.perf_counter()-t_start:.4f}s)")
                return {"text": content, "headlines": top_headlines, "source": "LIVE", "fetched_at": now}
            
        except Exception as e:
            print(f"[TRACE] News ingestion error for {query}: {e}")
            
        # 3. Defensive Fallback
        print(f"[TRACE] STEP 8: News ingestion complete (Fallback used)")
        text = self.fallback_news.get(transport_mode.lower(), "Normal operational conditions reported.")
        return {"text": text, "headlines": [], "source": "FALLBACK", "fetched_at": now}
