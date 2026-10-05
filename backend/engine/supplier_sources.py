"""
Live supplier discovery.

    Open Supply Hub   opensupplyhub.org - open, verified map of production facilities. Needs a free
                      API token in OSH_API_TOKEN (sign up, then Settings -> API token).
    Wikidata          real manufacturers by industry with headquarters coordinates. No key needed;
                      used whenever Open Supply Hub is not configured or fails.

Each source returns facilities in one shape:
    {id, name, lat, lon, iso, country, city, sector, source, url}
Results are cached per (category, country) for SUPPLIER_CACHE_HOURS (default 6).
"""
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import httpx

UA = {"User-Agent": "Supplychainer/2.0 (supplier intelligence; https://supplychainer.onrender.com)"}

# category -> Wikidata industries (P452) and keywords matched against Open Supply Hub sector names
CATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Electronics": {"wikidata": ["Q5358497", "Q2986369"], "osh": ["electronic"]},
    "Semiconductors": {"wikidata": ["Q2986369"], "osh": ["semiconductor", "electronic"]},
    "Automotive & parts": {"wikidata": ["Q190117"], "osh": ["automotive", "vehicle"]},
    "Chemicals": {"wikidata": ["Q207652"], "osh": ["chemical"]},
    "Pharmaceuticals": {"wikidata": ["Q507443"], "osh": ["pharma", "health"]},
    "Textiles & apparel": {"wikidata": ["Q607081", "Q11828862"], "osh": ["apparel", "textile"]},
    "Steel & metals": {"wikidata": ["Q2285982"], "osh": ["metal", "steel"]},
    "Machinery": {"wikidata": ["Q1957908"], "osh": ["machinery", "equipment"]},
    "Food & beverage": {"wikidata": ["Q540912"], "osh": ["food", "beverage", "agricultur"]},
}

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
OSH_BASE = os.getenv("OSH_API_BASE", "https://opensupplyhub.org/api")


class SupplierSourceError(RuntimeError):
    pass


def _point(wkt: str):
    m = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", wkt or "")
    return (float(m[2]), float(m[1])) if m else None


class WikidataSource:
    name = "Wikidata"
    key = "wikidata"

    def __init__(self):
        self._country_qids: Dict[str, Optional[str]] = {}

    @staticmethod
    def _sparql(query: str, timeout: float = 30) -> List[Dict[str, Any]]:
        try:
            r = httpx.get(WIKIDATA_SPARQL, params={"query": query}, timeout=timeout,
                          headers={**UA, "Accept": "application/sparql-results+json"})
            r.raise_for_status()
            return r.json()["results"]["bindings"]
        except Exception as e:
            raise SupplierSourceError(f"Wikidata query failed: {e or type(e).__name__}") from e

    def _country_qid(self, iso: str) -> Optional[str]:
        if iso not in self._country_qids:
            rows = self._sparql(f'SELECT ?c WHERE {{ ?c wdt:P297 "{iso}" . }} LIMIT 1', timeout=15)
            self._country_qids[iso] = rows[0]["c"]["value"].rsplit("/", 1)[-1] if rows else None
        return self._country_qids[iso]

    def fetch(self, category: str, iso: Optional[str], limit: int) -> List[Dict[str, Any]]:
        inds = " ".join(f"wd:{q}" for q in CATEGORIES[category]["wikidata"])
        country = ""
        if iso:
            qid = self._country_qid(iso)
            if not qid:
                raise SupplierSourceError(f"Unknown country code {iso}")
            country = f"; wdt:P17 wd:{qid}"
        # The inner select ranks by notability (number of Wikipedia sitelinks) before labels are
        # resolved; doing it in one flat query times out for country-filtered searches.
        query = f"""
        SELECT ?c ?cLabel ?coord ?iso ?hqLabel ?links WHERE {{
          {{ SELECT ?c ?coord ?hq ?links WHERE {{
              VALUES ?ind {{ {inds} }}
              ?c wdt:P452 ?ind ; wdt:P159 ?hq {country} .
              ?hq wdt:P625 ?coord .
              ?c wikibase:sitelinks ?links .
              FILTER NOT EXISTS {{ ?c wdt:P576 ?dissolved }}
            }} ORDER BY DESC(?links) LIMIT {limit * 3} }}
          OPTIONAL {{ ?c wdt:P17 ?cty . ?cty wdt:P297 ?iso . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}"""
        rows = sorted(self._sparql(query), key=lambda b: -int(b.get("links", {}).get("value", 0)))

        out, seen = [], set()
        for b in rows:
            qid = b["c"]["value"].rsplit("/", 1)[-1]
            name = b.get("cLabel", {}).get("value", "")
            pt = _point(b.get("coord", {}).get("value"))
            if qid in seen or not pt or re.fullmatch(r"Q\d+", name):
                continue
            seen.add(qid)
            out.append({"id": f"WD-{qid}", "name": name, "lat": pt[0], "lon": pt[1],
                        "iso": iso or b.get("iso", {}).get("value"), "city": b.get("hqLabel", {}).get("value"),
                        "sector": category, "source": self.name, "url": f"https://www.wikidata.org/wiki/{qid}",
                        "site": "Headquarters"})
            if len(out) >= limit:
                break
        return out


class OpenSupplyHubSource:
    name = "Open Supply Hub"
    key = "opensupplyhub"

    def __init__(self, token: str):
        self.token = token
        self._sectors: Optional[List[str]] = None

    def _client(self):
        return httpx.Client(base_url=OSH_BASE, timeout=25, headers={**UA, "Authorization": f"Token {self.token}"})

    def _sector_names(self, c, category) -> List[str]:
        if self._sectors is None:
            r = c.get("/sectors/")
            r.raise_for_status()
            data = r.json()
            # Either a flat list of names or grouped [{group_name, sectors: [...]}]
            names = []
            for item in data if isinstance(data, list) else data.get("results", []):
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    names += item.get("sectors") or ([item["name"]] if item.get("name") else [])
            self._sectors = names
        keys = CATEGORIES[category]["osh"]
        return [s for s in self._sectors if any(k in s.lower() for k in keys)][:6]

    def fetch(self, category: str, iso: Optional[str], limit: int) -> List[Dict[str, Any]]:
        try:
            with self._client() as c:
                params = [("pageSize", str(limit)), ("page", "1")]
                params += [("sectors", s) for s in self._sector_names(c, category)]
                if iso:
                    params.append(("countries", iso))
                r = c.get("/facilities/", params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            raise SupplierSourceError(f"Open Supply Hub request failed: {e}") from e

        out = []
        for f in data.get("features", []):
            props, geom = f.get("properties", {}), f.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            if coords[0] is None:
                continue
            os_id = props.get("os_id") or f.get("id")
            sector = props.get("sector")
            out.append({"id": f"OSH-{os_id}", "name": props.get("name") or os_id,
                        "lat": float(coords[1]), "lon": float(coords[0]),
                        "iso": props.get("country_code"), "city": props.get("address"),
                        "sector": ", ".join(sector) if isinstance(sector, list) else (sector or category),
                        "source": self.name, "url": f"https://opensupplyhub.org/facility/{os_id}",
                        "site": "Production facility"})
        return out


class SupplierDirectory:
    def __init__(self):
        token = os.getenv("OSH_API_TOKEN", "").strip()
        self.primary = OpenSupplyHubSource(token) if token else None
        self.fallback = WikidataSource()
        self.ttl = float(os.getenv("SUPPLIER_CACHE_HOURS", "6")) * 3600
        self._cache: Dict[tuple, tuple] = {}
        self._lock = threading.Lock()

    def categories(self) -> List[str]:
        return list(CATEGORIES)

    def info(self) -> Dict[str, Any]:
        return {"primary": self.primary.name if self.primary else None, "fallback": self.fallback.name,
                "open_supply_hub_configured": self.primary is not None}

    def find(self, category: str, iso: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        if category not in CATEGORIES:
            raise SupplierSourceError(f"Unknown category {category}")
        iso = (iso or "").upper() or None
        key = (category, iso, limit)
        with self._lock:
            hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return {**hit[1], "cached": True}

        notes, facilities, used = [], [], None
        for src in filter(None, (self.primary, self.fallback)):
            try:
                facilities = src.fetch(category, iso, limit)
                used = src
                if facilities:
                    break
                notes.append(f"{src.name} returned no {category.lower()} suppliers"
                             + (f" in {iso}." if iso else "."))
            except SupplierSourceError as e:
                notes.append(str(e))
        if used is None:
            raise SupplierSourceError(" ".join(notes) or "No supplier source available.")
        result = {"facilities": facilities, "source": used.name, "notes": notes,
                  "fetched_at": time.time(), "cached": False}
        with self._lock:
            self._cache[key] = (time.time(), result)
        return result
