"""
Real-world port and airport registry, built from live public datasets.

    ports     UN/LOCODE code list (UNECE) - every location flagged as a port, with coordinates
              (~11,800). Enriched with harbour size, container handling and max draft from the
              NGA World Port Index where the two match on UN/LOCODE.
    airports  OurAirports (refreshed nightly) - large and medium airports with scheduled service
              and an IATA code (~3,200).

The routing graph keeps its curated core of ~444 hubs. A real port or airport is attached to that
graph only when a request uses it, through short feeder legs to the nearest core hubs (sea feeders
to nearby core ports, air feeders to nearby core airports, road feeders to core hubs in the same
country). A record within a few km of a core hub is treated as an alias of that hub.

A snapshot ships in backend/data/world_registry.json so the app works offline. At startup a
background thread rebuilds the registry from the live sources when the local copy is older than
WORLD_REFRESH_DAYS (default 7) and writes it to the cache directory.
"""
import csv
import io
import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
BUNDLED_SNAPSHOT = os.path.join(DATA_DIR, "world_registry.json")

SOURCES = {
    "unlocode": "https://raw.githubusercontent.com/datasets/un-locode/main/data/code-list.csv",
    "wpi": "https://msi.nga.mil/api/publications/world-port-index?output=json",
    "airports": "https://davidmegginson.github.io/ourairports-data/airports.csv",
    "countries": "https://davidmegginson.github.io/ourairports-data/countries.csv",
}
SOURCE_LABELS = {
    "unlocode": "UN/LOCODE (UNECE)", "wpi": "NGA World Port Index",
    "airports": "OurAirports", "countries": "OurAirports",
}

# Core-hub country names that differ from the ISO short names in the datasets.
COUNTRY_ALIASES = {"USA": "US", "UK": "GB", "UAE": "AE", "Ivory Coast": "CI", "Somaliland": "SO"}

ALIAS_RADIUS_KM = 25          # a real record this close to a core hub *is* that hub
SEA_FEEDERS, AIR_FEEDERS = 3, 3
AIR_FEEDER_RADIUS_KM = 1200   # also connect every core airport this close (max 5 air feeders)
ROAD_FEEDER_RADIUS_KM = 400
HARBOR_IMPORTANCE = {"L": 8, "M": 6, "S": 4, "V": 3}
# The Caspian Sea is landlocked: its ports only connect by sea to other Caspian ports.
CASPIAN = (36.5, 47.5, 46.5, 55.5)


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _basin(lat, lon):
    s, n, w, e = CASPIAN
    return "caspian" if s <= lat <= n and w <= lon <= e else "ocean"


def _unloc_coords(text: str) -> Optional[Tuple[float, float]]:
    """'5155N 00430E' -> (51.9167, 4.5)"""
    m = re.match(r"^(\d{2})(\d{2})([NS])\s+(\d{3})(\d{2})([EW])$", (text or "").strip())
    if not m:
        return None
    lat = int(m[1]) + int(m[2]) / 60
    lon = int(m[4]) + int(m[5]) / 60
    lat, lon = (-lat if m[3] == "S" else lat), (-lon if m[6] == "W" else lon)
    if abs(lat) > 90 or abs(lon) > 180:
        return None
    return round(lat, 4), round(lon, 4)


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------- building
def build_registry(timeout: float = 60.0) -> Dict[str, Any]:
    """Downloads every source and returns a fresh registry document. Raises on failure."""
    with httpx.Client(timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "Supplychainer/2.0 (world registry refresh)"}) as client:
        def get(key):
            r = client.get(SOURCES[key])
            r.raise_for_status()
            return r

        countries = {row["code"]: row["name"] for row in csv.DictReader(io.StringIO(get("countries").text))}
        unloc_rows = list(csv.DictReader(io.StringIO(get("unlocode").text)))
        try:
            wpi_ports = get("wpi").json().get("ports", [])
            wpi_ok = True
        except Exception as e:  # the WPI service is flaky; ports still come from UN/LOCODE
            print(f"[WORLD] World Port Index unavailable ({e}); continuing without harbour details.")
            wpi_ports, wpi_ok = [], False
        airport_rows = list(csv.DictReader(io.StringIO(get("airports").text)))

    wpi = {}
    for p in wpi_ports:
        code = (p.get("unloCode") or "").replace(" ", "").upper()
        if len(code) == 5:
            wpi[code] = p

    ports, seen = [], set()
    for row in unloc_rows:
        if (row.get("Function") or "")[:1] != "1":
            continue
        code = f"{row['Country']}{row['Location']}".upper()
        if code in seen:
            continue
        w = wpi.get(code)
        coords = (round(w["ycoord"], 4), round(w["xcoord"], 4)) if w and w.get("ycoord") is not None \
            else _unloc_coords(row.get("Coordinates"))
        if not coords:
            continue
        seen.add(code)
        size = (w or {}).get("harborSize")
        approved = row.get("Status") in ("AA", "AC", "AF", "AI", "AS")
        ports.append({
            "id": f"UNLOC-{code}", "type": "port", "display_name": row.get("NameWoDiacritics") or row["Name"],
            "iso": row["Country"], "country": countries.get(row["Country"], row["Country"]),
            "lat": coords[0], "lon": coords[1], "unlocode": code,
            "harbor_size": size, "container": (w or {}).get("loContainer") == "Y" or None,
            "max_draft_m": _num((w or {}).get("maxVesselDraft")),
            "water_body": ((w or {}).get("dodWaterBody") or "").split(";")[0].strip() or None,
            "importance": HARBOR_IMPORTANCE.get(size, 3 if approved else 2),
            "modes": ["sea", "road"],
        })

    airports = []
    for row in airport_rows:
        if row["type"] not in ("large_airport", "medium_airport") or row["scheduled_service"] != "yes" \
                or not row["iata_code"]:
            continue
        lat, lon = _num(row["latitude_deg"]), _num(row["longitude_deg"])
        if lat is None or lon is None:
            continue
        large = row["type"] == "large_airport"
        airports.append({
            "id": f"IATA-{row['iata_code']}", "type": "airport", "display_name": row["name"],
            "iso": row["iso_country"], "country": countries.get(row["iso_country"], row["iso_country"]),
            "city": row["municipality"] or None, "lat": round(lat, 4), "lon": round(lon, 4),
            "iata": row["iata_code"], "icao": row["icao_code"] or row["gps_code"] or None,
            "size": "large" if large else "medium", "importance": 7 if large else 4,
            "modes": ["air", "road"],
        })

    now = time.time()
    used = ["unlocode", "airports", "countries"] + (["wpi"] if wpi_ok else [])
    return {
        "generated_at": now,
        "sources": [{"key": k, "name": SOURCE_LABELS[k], "url": SOURCES[k]} for k in used],
        "countries": countries,
        "ports": ports,
        "airports": airports,
    }


# ---------------------------------------------------------------------------- registry
class WorldRegistry:
    def __init__(self, core_hubs: List[Dict[str, Any]], cache_path: Optional[str] = None):
        self.core = core_hubs
        self.cache_path = cache_path or os.getenv("WORLD_REGISTRY_CACHE") or os.path.join(
            os.path.dirname(os.path.abspath(os.getenv("SUPPLYCHAINER_DB", BUNDLED_SNAPSHOT))), "world_registry.cache.json")
        self.refresh_days = float(os.getenv("WORLD_REFRESH_DAYS", "7"))
        self._lock = threading.Lock()
        self.refreshing = False
        self.last_error: Optional[str] = None
        self.doc: Dict[str, Any] = {"generated_at": 0, "sources": [], "countries": {}, "ports": [], "airports": []}
        self.records: Dict[str, Dict[str, Any]] = {}
        for path in (self.cache_path, BUNDLED_SNAPSHOT):
            doc = self._read(path)
            if doc and doc.get("generated_at", 0) > self.doc["generated_at"]:
                self.doc = doc
        self._index()

    @staticmethod
    def _read(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    # ------------------------------------------------------------------ indexing
    def _index(self):
        core_iso = {}
        for h in self.core:
            core_iso[h["id"]] = COUNTRY_ALIASES.get(h["country"]) or self._iso_for_name(h["country"])
        self.core_iso = core_iso
        self.core_ports = [h for h in self.core if h["type"] == "port" and "sea" in h["modes"]]
        self.core_airports = [h for h in self.core if "air" in h["modes"]]
        self.core_road = [h for h in self.core if "road" in h["modes"]]

        records = {}
        for rec in self.doc["ports"] + self.doc["airports"]:
            rec = dict(rec)
            pool = self.core_ports if rec["type"] == "port" else self.core_airports
            near = min(pool, key=lambda h: haversine(rec["lat"], rec["lon"], h["lat"], h["lon"]), default=None)
            if near and haversine(rec["lat"], rec["lon"], near["lat"], near["lon"]) <= ALIAS_RADIUS_KM:
                rec["core_alias"] = near["id"]
            records[rec["id"]] = rec
        gateways = [r for r in records.values() if r["type"] == "port"
                    and (r.get("core_alias") or r.get("harbor_size") in ("L", "M", "S"))]
        with self._lock:
            self.records = records
            self.gateways = gateways
            self._search_rows = [(self._norm(" ".join(filter(None, [
                r["display_name"], r.get("city"), r["country"], r.get("unlocode"), r.get("iata"), r.get("icao")]))), r)
                for r in records.values()]

    def _iso_for_name(self, name):
        for iso, n in self.doc.get("countries", {}).items():
            if n == name:
                return iso
        return None

    @staticmethod
    def _norm(s):
        return re.sub(r"\s+", " ", (s or "").lower()).strip()

    # ------------------------------------------------------------------ queries
    def get(self, rid: str) -> Optional[Dict[str, Any]]:
        return self.records.get(rid)

    def is_world_id(self, rid: str) -> bool:
        return isinstance(rid, str) and (rid.startswith("UNLOC-") or rid.startswith("IATA-"))

    def search(self, q: str, limit: int = 15) -> List[Dict[str, Any]]:
        q = self._norm(q)
        if not q:
            return []
        hits = []
        for text, r in self._search_rows:
            if q not in text:
                continue
            name = self._norm(r["display_name"])
            code_hit = q.upper() in (r.get("unlocode"), r.get("iata"), r.get("icao"))
            rank = 0 if code_hit else 1 if name.startswith(q) else 2 if any(w.startswith(q) for w in name.split()) else 3
            hits.append((rank, -r["importance"], r["display_name"], r))
        hits.sort(key=lambda x: x[:3])
        return [self.public(r) for *_, r in hits[:limit]]

    def nearest_port(self, lat: float, lon: float) -> Optional[Tuple[Dict[str, Any], float]]:
        """Nearest real *seaport* for export cargo: a port in the World Port Index or one of the core
        ports. UN/LOCODE also flags river landings and marinas as ports, which are no export gateway."""
        best = None
        for r in self.gateways:
            d = haversine(lat, lon, r["lat"], r["lon"])
            if best is None or d < best[1]:
                best = (r, d)
        return best

    def nearest_land_hub(self, lat: float, lon: float, iso: Optional[str],
                         max_km: float = 500) -> Optional[Tuple[Dict[str, Any], float]]:
        """Nearest core road hub in the same country: lets the routing engine pick the export port."""
        if not iso:
            return None
        best = None
        for h in self.core_road:
            if self.core_iso.get(h["id"]) != iso or h["type"] == "choke_point":
                continue
            d = haversine(lat, lon, h["lat"], h["lon"])
            if d <= max_km and (best is None or d < best[1]):
                best = (h, d)
        return best

    @staticmethod
    def public(r: Dict[str, Any]) -> Dict[str, Any]:
        """Shape compatible with the core hub search results the UI already renders."""
        out = {k: v for k, v in r.items() if k not in ("iso",)}
        out["source"] = "world"
        out["parent_city"] = r.get("city") or r["display_name"]
        return out

    # ------------------------------------------------------------------ graph attachment
    def feeders(self, rec: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """(core_hub_id, mode, distance_km) links that connect a real record to the core graph."""
        lat, lon = rec["lat"], rec["lon"]
        dist = lambda h: haversine(lat, lon, h["lat"], h["lon"])
        out = []
        if rec["type"] == "port":
            basin = _basin(lat, lon)
            pool = sorted((h for h in self.core_ports if _basin(h["lat"], h["lon"]) == basin), key=dist)
            out += [(h["id"], "sea", dist(h)) for h in pool[:SEA_FEEDERS]]
        else:
            pool = sorted(self.core_airports, key=dist)
            near = [h for h in pool[:5] if dist(h) <= AIR_FEEDER_RADIUS_KM]
            chosen = near if len(near) >= AIR_FEEDERS else pool[:AIR_FEEDERS]
            out += [(h["id"], "air", dist(h)) for h in chosen]
        iso = rec.get("iso")
        road = sorted((h for h in self.core_road if self.core_iso.get(h["id"]) == iso and dist(h) <= ROAD_FEEDER_RADIUS_KM),
                      key=dist)
        out += [(h["id"], "road", dist(h)) for h in road[:2]]
        return out

    # ------------------------------------------------------------------ status + refresh
    def status(self) -> Dict[str, Any]:
        return {"ports": len(self.doc["ports"]), "airports": len(self.doc["airports"]),
                "countries": len({r["iso"] for r in self.doc["ports"] + self.doc["airports"]}),
                "generated_at": self.doc["generated_at"], "sources": self.doc["sources"],
                "refreshing": self.refreshing, "last_error": self.last_error}

    def is_stale(self) -> bool:
        return time.time() - self.doc.get("generated_at", 0) > self.refresh_days * 86400

    def refresh(self) -> bool:
        self.refreshing = True
        try:
            doc = build_registry()
            if len(doc["ports"]) < 1000 or len(doc["airports"]) < 500:
                raise ValueError(f"suspiciously small download ({len(doc['ports'])} ports, {len(doc['airports'])} airports)")
            self.doc = doc
            self._index()
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(doc, f, separators=(",", ":"))
            except OSError as e:
                print(f"[WORLD] Could not write cache: {e}")
            self.last_error = None
            print(f"[WORLD] Registry refreshed: {len(doc['ports'])} ports, {len(doc['airports'])} airports.")
            return True
        except Exception as e:
            self.last_error = str(e)
            print(f"[WORLD] Refresh failed, keeping the snapshot: {e}")
            return False
        finally:
            self.refreshing = False

    def refresh_in_background_if_stale(self):
        if os.getenv("WORLD_REFRESH", "true").lower() == "false" or not self.is_stale() or self.refreshing:
            return
        threading.Thread(target=self.refresh, daemon=True, name="world-registry-refresh").start()


if __name__ == "__main__":
    # python -m backend.engine.world_registry   -> rebuilds the bundled snapshot from the live sources
    doc = build_registry()
    with open(BUNDLED_SNAPSHOT, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"Wrote {BUNDLED_SNAPSHOT}: {len(doc['ports'])} ports, {len(doc['airports'])} airports.")
