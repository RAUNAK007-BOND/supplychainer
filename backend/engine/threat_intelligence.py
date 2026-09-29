import numpy as np
import joblib
import os
import re
import math
import json
import threading
from itertools import combinations
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# Load Production Artifacts
MODEL_PATH = "./Execution/risk_model.pkl"            # shipped p85 quantile model
MODEL_P50_PATH = "./Execution/risk_model_p50.pkl"    # Code/train_quantile_band.py
MODEL_P95_PATH = "./Execution/risk_model_p95.pkl"    # Code/train_quantile_band.py
ENCODER_PATH = "./Execution/label_encoders.pkl"
NLP_ANCHORS_PATH = "./Execution/nlp_anchors.pt"
CALIBRATION_PATH = "./Execution/calibration_profiles.json"
SHAP_BACKGROUND_PATH = "./Execution/shap_background.json"

FEATURES = ["Leg_Type", "Origin_Node", "Destination_Node", "Transport_Mode", "Condition_Flag", "NLP_Severity_Score"]
FEATURE_LABELS = {
    "Leg_Type": "Leg type",
    "Origin_Node": "Origin dwell profile",
    "Destination_Node": "Destination dwell profile",
    "Transport_Mode": "Transport mode",
    "Condition_Flag": "Weather condition",
    "NLP_Severity_Score": "News severity (NLP+CARF)",
}

# Canonical hubs that correspond 1:1 to a node the model was trained on.
DIRECT_MODEL_NODES = {
    "PORT-SHANGHAI": "Shanghai Port", "PORT-SINGAPORE": "Singapore Port", "PORT-MUMBAI": "Mumbai Port",
    "PORT-ROTTERDAM": "Rotterdam Port", "PORT-LOSANGELES": "Los Angeles Port", "PORT-LONGBEACH": "Los Angeles Port",
    "PORT-NEWYORK": "New York Port", "PORT-HOUSTON": "Houston Port", "PORT-SEATTLE": "Seattle Port",
    "CHOKE-SUEZ": "Suez Canal", "CHOKE-PANAMA": "Suez Canal",  # canal transit queue analogue
    "AIR-DUBAI": "Dubai Logistics Hub", "AIR-DELHI": "Delhi Air Cargo",
    "AIR-ATLANTA": "Atlanta Air Hub", "RAIL-CHICAGO": "Chicago Rail Hub", "HUB-CHICAGO": "Chicago Rail Hub",
    "RAIL-HOUSTON": "Houston Port", "HUB-DALLAS": "Dallas Corridor",
}

# Every other hub borrows the dwell profile of the geographically nearest anchor of the same mode
# (the "nearest LPI tier" approach described in REAL_DATA_TRUTH_AUDIT.md). Previously any unknown
# hub silently fell back to encoder class 0 ("Atlanta Air Hub") regardless of mode.
MODE_ANCHORS = {
    "sea": {"Seattle Port": (47.60, -122.34), "Los Angeles Port": (33.74, -118.27), "New York Port": (40.68, -74.04),
            "Rotterdam Port": (51.95, 4.14), "Mumbai Port": (18.95, 72.84), "Singapore Port": (1.26, 103.84),
            "Shanghai Port": (31.23, 121.49)},
    "rail": {"Chicago Rail Hub": (41.88, -87.63), "Houston Port": (29.76, -95.36),
             "St. Louis Hub": (38.63, -90.20), "Dallas Corridor": (32.78, -96.80)},
    "air": {"Atlanta Air Hub": (33.64, -84.43), "Delhi Air Cargo": (28.56, 77.10), "Dubai Logistics Hub": (25.25, 55.36)},
}


def _haversine(lat1, lon1, lat2, lon2):
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class ThreatIntelligencePredictor:
    """
    Supplychainer Quantile ML Decision Brain.
    V4: p50/p85/p95 confidence band, mode-aware hub profiling, exact Shapley explanations.
    """
    QUANTILES = ("p50", "p85", "p95")

    def __init__(self, lazy_load=False):
        self.is_trained = False
        self.model = None
        self.models = {}
        self.encoders = None
        self.profiles = {}
        self.background = None
        self._cache: Dict[tuple, Dict[str, float]] = {}
        self._explain_cache: Dict[tuple, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Legacy city-name aliases (kept for callers that pass city names instead of hub IDs)
        self.hub_map = {
            "Seattle": "Seattle Port", "Los Angeles": "Los Angeles Port", "Houston": "Houston Port",
            "Chicago": "Chicago Rail Hub", "St. Louis": "St. Louis Hub", "Atlanta": "Atlanta Air Hub",
            "Dallas": "Dallas Corridor", "New York": "New York Port", "Mumbai": "Mumbai Port",
            "Delhi": "Delhi Air Cargo", "Shanghai": "Shanghai Port", "Singapore": "Singapore Port",
            "Rotterdam": "Rotterdam Port", "Dubai": "Dubai Logistics Hub",
        }

        if not lazy_load:
            self.warmup()

    def warmup(self):
        if self.is_trained: return
        print("[PREDICTOR] Starting warmup...")
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
            print(f"CRITICAL: Production models missing. Running in deterministic fallback mode.")
            return

        # 1. Load ML Core (p85 is the shipped model; p50/p95 are optional companions)
        self.model = joblib.load(MODEL_PATH)
        self.models = {"p85": self.model}
        for tag, path in (("p50", MODEL_P50_PATH), ("p95", MODEL_P95_PATH)):
            if os.path.exists(path):
                self.models[tag] = joblib.load(path)
            else:
                print(f"WARNING: {path} missing — {tag} will be derived from p85. Run Code/train_quantile_band.py.")
        self.encoders = joblib.load(ENCODER_PATH)
        self.is_trained = True

        # 2. Load Statistically Defensible Calibration Profiles
        if os.path.exists(CALIBRATION_PATH):
            with open(CALIBRATION_PATH, 'r') as f:
                self.profiles = json.load(f)
            print(f"Calibration Layer: Loaded {len(self.profiles)} mode profiles from historical p5/p95 analysis.")
        else:
            print("WARNING: Calibration profiles missing. Using defensive fallbacks.")
            self.profiles = {}

        if os.path.exists(SHAP_BACKGROUND_PATH) and os.path.getsize(SHAP_BACKGROUND_PATH) > 0:
            with open(SHAP_BACKGROUND_PATH) as f:
                self.background = np.array(json.load(f)["rows"], dtype=float)

        print(f"Supplychainer V4 Brain Loaded: quantiles={sorted(self.models)}.")

    # ------------------------------------------------------------------ feature mapping
    def model_node_for(self, hub: Optional[Dict[str, Any]], mode: str, role: str) -> str:
        """Maps a canonical hub to the training-set node whose dwell profile it should borrow."""
        mode = mode.lower()
        if mode == "road":
            # Road legs were only ever trained as Regional Hub -> Local Terminal.
            return "Regional Hub" if role == "origin" else "Local Terminal"
        if hub is None:
            return "Regional Hub" if role == "origin" else "Local Terminal"
        direct = DIRECT_MODEL_NODES.get(hub.get("physical_id") or hub.get("id"))
        if direct:
            return direct
        anchors = MODE_ANCHORS.get(mode, {})
        if not anchors or hub.get("lat") is None:
            return "Regional Hub" if role == "origin" else "Local Terminal"
        return min(anchors, key=lambda a: _haversine(hub["lat"], hub["lon"], *anchors[a]))

    def _encode(self, value: str, key: str) -> int:
        encoder = self.encoders[key]
        classes = list(encoder.classes_)
        if key in ("Origin_Node", "Destination_Node"):
            value = self.hub_map.get(value, value)
            if value not in classes:
                value = "Regional Hub" if key == "Origin_Node" else "Local Terminal"
        if key == "Transport_Mode":
            value = str(value).lower()
        if value not in classes:
            value = classes[0]
        return int(encoder.transform([value])[0])

    def _row(self, key: tuple) -> List[float]:
        o, d, mode, leg, cond, nlp = key
        return [self._encode(leg, "Leg_Type"), self._encode(o, "Origin_Node"), self._encode(d, "Destination_Node"),
                self._encode(mode, "Transport_Mode"), self._encode(cond, "Condition_Flag"), float(nlp)]

    @staticmethod
    def make_key(origin_node: str, dest_node: str, mode: str, nlp_score: float, condition: str = "Clear") -> tuple:
        mode = mode.lower()
        leg = "Global_Freight" if mode in ("sea", "air") else "Last_Mile"
        return (origin_node, dest_node, mode, leg, condition, round(float(nlp_score), 2))

    # ------------------------------------------------------------------ inference
    def _calibrate(self, mode: str, raw: Dict[str, float]) -> Dict[str, Any]:
        profile = self.profiles.get(mode, {"floor": 0.0, "cap": 240.0, "p95_observed": 240.0})
        floor, cap = profile["floor"], profile["cap"]
        tail_cap = max(cap, profile.get("p95_observed", cap))
        p85 = min(max(raw["p85"], floor), cap)
        p50 = min(max(raw["p50"], floor), cap)
        p95 = min(max(raw["p95"], floor), tail_cap)
        # Independently trained quantile models can cross; enforce p50 <= p85 <= p95.
        p50, p95 = min(p50, p85), max(p95, p85)

        reason = "Quantile Disruption Prediction (p85 Risk)"
        if raw["p85"] < floor:
            reason = f"Baseline Operational Friction (Historical p5: {floor}h)"
        elif raw["p85"] > cap:
            reason = f"Operational Cap Applied (Historical p95 Bound: {cap}h)"
        return {"p50": round(p50, 2), "p85": round(p85, 2), "p95": round(p95, 2),
                "raw_p85": round(raw["p85"], 2), "calibration_reason": reason}

    def predict_band_many(self, keys: List[tuple]) -> List[Dict[str, Any]]:
        """Vectorised p50/p85/p95 inference with a memo cache (keys from make_key)."""
        if not self.is_trained:
            priors = {"road": 2.5, "sea": 48.0, "air": 12.0, "rail": 18.0}
            out = []
            for k in keys:
                p = priors.get(k[2], 12.0)
                out.append({"p50": round(p * 0.6, 2), "p85": p, "p95": round(p * 1.6, 2), "raw_p85": p,
                            "calibration_reason": "Deterministic Operational Prior (Engine Warming)"})
            return out

        missing = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if missing:
            X = pd.DataFrame([self._row(k) for k in missing], columns=FEATURES)
            raw = {tag: m.predict(X) for tag, m in self.models.items()}
            if "p50" not in raw: raw["p50"] = raw["p85"] * 0.55
            if "p95" not in raw: raw["p95"] = raw["p85"] * 1.5
            with self._lock:
                for i, k in enumerate(missing):
                    self._cache[k] = self._calibrate(k[2], {t: float(raw[t][i]) for t in self.QUANTILES})
        return [self._cache[k] for k in keys]

    def predict_band(self, origin_node: str, dest_node: str, mode: str, nlp_score: float,
                     condition: str = "Clear") -> Dict[str, Any]:
        return self.predict_band_many([self.make_key(origin_node, dest_node, mode, nlp_score, condition)])[0]

    def predict_worst_case_delay(self, origin: str, destination: str, transport_mode: str,
                                 leg_type: str = "Global_Freight", condition_flag: str = "Clear",
                                 nlp_score: float = 0.0) -> Dict[str, Any]:
        """Stage 4: p85 Quantile Prediction (backwards-compatible single-point API)."""
        band = self.predict_band_many([(origin, destination, transport_mode.lower(), leg_type,
                                        condition_flag, round(float(nlp_score), 2))])[0]
        profile = self.profiles.get(transport_mode.lower(), {"floor": 0.0})
        return {
            "raw_model_prediction": band["raw_p85"],
            "calibrated_delay": band["p85"],
            "baseline_systemic_friction": profile.get("floor", 0.0),
            "final_delay_presented": band["p85"],
            "band": {q: band[q] for q in self.QUANTILES},
            "calibration_reason": band["calibration_reason"],
            "p_quantile": 0.85,
            "is_defensible": True,
        }

    # ------------------------------------------------------------------ explainability
    def explain(self, key: tuple, quantile: str = "p85") -> Optional[Dict[str, Any]]:
        """
        Exact interventional Shapley values for one prediction.

        With only 6 features we can enumerate all 2^6 coalitions exactly (no KernelSHAP
        sampling): v(S) = mean over a background sample of f(x_S, b_~S). The contributions sum
        exactly to f(x) - E[f(background)], in hours of predicted delay.
        """
        if not self.is_trained or self.background is None:
            return None
        model = self.models.get(quantile, self.model)
        ck = (key, quantile)
        if ck in self._explain_cache:
            return self._explain_cache[ck]

        x = np.array(self._row(key), dtype=float)
        n, bg = len(FEATURES), self.background
        subsets = [s for r in range(n + 1) for s in combinations(range(n), r)]
        blocks = []
        for s in subsets:
            b = bg.copy()
            if s: b[:, list(s)] = x[list(s)]
            blocks.append(b)
        preds = model.predict(pd.DataFrame(np.vstack(blocks), columns=FEATURES)).reshape(len(subsets), len(bg))
        value = {s: float(preds[i].mean()) for i, s in enumerate(subsets)}

        phi = np.zeros(n)
        for i in range(n):
            for s in subsets:
                if i in s: continue
                w = math.factorial(len(s)) * math.factorial(n - len(s) - 1) / math.factorial(n)
                phi[i] += w * (value[tuple(sorted(s + (i,)))] - value[s])

        shown = {"Leg_Type": key[3], "Origin_Node": key[0], "Destination_Node": key[1],
                 "Transport_Mode": key[2], "Condition_Flag": key[4], "NLP_Severity_Score": key[5]}
        contributions = sorted(({"feature": FEATURE_LABELS[f], "value": shown[f], "hours": round(float(phi[i]), 2)}
                                for i, f in enumerate(FEATURES)), key=lambda c: -abs(c["hours"]))
        result = {"method": "exact interventional Shapley (64 coalitions)", "quantile": quantile,
                  "base_value_h": round(value[()], 2), "model_output_h": round(value[tuple(range(n))], 2),
                  "contributions": contributions}
        self._explain_cache[ck] = result
        return result

    def model_info(self) -> Dict[str, Any]:
        info = {"trained": self.is_trained, "quantiles": sorted(self.models), "features": FEATURES}
        if self.is_trained:
            info["feature_importance_p85"] = {FEATURE_LABELS[f]: round(float(v), 3)
                                              for f, v in zip(FEATURES, self.model.feature_importances_)}
            info["calibration_profiles"] = self.profiles
        report = "./Execution/quantile_band_report.json"
        if os.path.exists(report):
            with open(report) as f:
                info["holdout_coverage"] = json.load(f)
        return info


class ContrastiveNLPEngine:
    """Stage 2: PRODUCTION Contrastive NLP Brain."""
    def __init__(self, lazy_load=False):
        self._ready = False
        # Margins at or below this are indistinguishable from routine operational news.
        self.noise_floor = 0.04
        # Scales the disaster-vs-safe cosine margin into a 0-1 severity (matches Code/nlp_engine.py).
        self.calibration_multiplier = 3.5
        if not lazy_load:
            self.warmup()

    @property
    def ready(self) -> bool:
        return self._ready

    def warmup(self):
        if self._ready: return
        if os.getenv("NLP_ENABLED", "true").lower() == "false":
            # Small hosts (e.g. 512 MB free tiers) can't fit PyTorch + the embedding model.
            print("[NLP ENGINE] Disabled by NLP_ENABLED=false; live news uses keyword scoring.")
            return
        print("[NLP ENGINE] Starting warmup...")
        try:
            import torch
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.util = util
            if os.path.exists(NLP_ANCHORS_PATH):
                # The anchors were saved from a CUDA session; without map_location they fail to
                # load on every CPU-only machine and the whole NLP stage silently scores 0.
                anchors = torch.load(NLP_ANCHORS_PATH, map_location="cpu")
                self.disaster_matrix = anchors["disaster_matrix"]
                self.safe_matrix = anchors["safe_matrix"]
                self._ready = True
                print(f"NLP Brain: Loaded Historical Anchor Matrix.")
            else:
                self._ready = False
        except Exception as e:
            print(f"[NLP ENGINE] Warmup failed: {e}")
            self._ready = False

    @staticmethod
    def _chunks(news_text: str) -> List[str]:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\s\|\s", news_text) if p.strip()]
        chunks = []
        for p in parts:
            chunks.extend(p[i:i + 256] for i in range(0, len(p), 256))
        return chunks or [news_text]

    def get_semantic_score(self, news_text: str) -> float:
        if not self._ready: return 0.0
        if not news_text or len(news_text.strip()) < 5: return 0.0
        chunks = self._chunks(news_text)
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        d_scores = self.util.cos_sim(chunk_embeddings, self.disaster_matrix).cpu().numpy()
        s_scores = self.util.cos_sim(chunk_embeddings, self.safe_matrix).cpu().numpy()
        # Per-chunk contrast, max-pooled: one alarming headline must not be diluted by calm ones.
        margin = float(np.max(d_scores.max(axis=1) - s_scores.max(axis=1)))
        if margin <= self.noise_floor: return 0.0
        return float(min(1.0, margin * self.calibration_multiplier))


THREAT_TAXONOMY = {
    "cyber": ["cyber", "cyberattack", "ransomware", "hack", "hacked", "malware"],
    "geopolitical": ["war", "conflict", "missile", "missiles", "attack", "attacks", "military", "naval", "blockade",
                     "sanction", "sanctions", "embargo", "pirate", "pirates", "piracy", "hijack", "hijacked",
                     "houthi", "escalation", "tension", "tensions", "security", "seized"],
    "labor": ["strike", "strikes", "union", "walkout", "picket", "labor", "labour", "protest", "protests", "workers"],
    "weather": ["storm", "storms", "hurricane", "typhoon", "cyclone", "flood", "floods", "flooding", "monsoon",
                "snow", "blizzard", "fog", "earthquake", "tsunami", "wildfire", "heatwave", "rain", "weather"],
    "infrastructure": ["grounding", "grounded", "aground", "collapse", "collapsed", "derail", "derailment",
                       "derailed", "outage", "closure", "closed", "blocked", "blockage", "accident", "fire",
                       "explosion", "maintenance", "salvage", "damage", "damaged"],
    "congestion": ["congestion", "congested", "backlog", "backlogs", "queue", "queues", "surge", "bottleneck",
                   "capacity", "berthing", "dwell", "delays", "delayed"],
}


def classify_threat(text: str) -> Dict[str, Any]:
    """Categorises a disruption by type (not just magnitude) so downstream logic can react differently."""
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    hits = {cat: sorted(tokens.intersection(words)) for cat, words in THREAT_TAXONOMY.items()}
    hits = {c: h for c, h in hits.items() if h}
    if not hits:
        return {"category": "operational", "evidence": []}
    # Ties resolve in taxonomy order (most specific class first).
    best = max(hits, key=lambda c: (len(hits[c]), -list(THREAT_TAXONOMY).index(c)))
    return {"category": best, "evidence": hits[best]}


def keyword_severity(text: str) -> float:
    """Coarse 0-1 severity from disruption keywords; used only when the embedding model is disabled."""
    tokens = set(re.findall(r"[a-z]+", (text or "").lower()))
    hits = sum(len(tokens & set(words)) for words in THREAT_TAXONOMY.values())
    return round(min(1.0, 0.3 * hits), 3)


class CARFFilter:
    """
    Stage 3: TRUE CARF (Context-Aware Relevance Filter).

    Enforced symmetrically for all four modes: news that is clearly about a *different* mode
    (and never mentions this one) is irrelevant; mode-matched news or area-wide events
    (floods, war, cyberattacks...) pass through; news that names no mode at all is attenuated.
    """
    AREA_WIDE = {"flood", "floods", "flooding", "earthquake", "tsunami", "war", "cyclone", "hurricane", "typhoon",
                 "pandemic", "lockdown", "curfew", "blackout", "cyberattack", "ransomware", "wildfire", "monsoon",
                 "sanctions", "embargo", "shutdown"}
    UNSPECIFIED_ATTENUATION = 0.5

    def __init__(self):
        self.relevance_map = {
            "air": {"airport", "airports", "flight", "flights", "airspace", "aviation", "aircraft", "airline",
                    "airlines", "runway", "air", "freighter", "sky"},
            "sea": {"port", "ports", "seaport", "vessel", "vessels", "ship", "ships", "shipping", "canal", "ocean",
                    "maritime", "dock", "docks", "berth", "berthing", "strait", "naval", "tanker", "harbor",
                    "harbour", "transshipment", "container", "sea", "pirates"},
            "rail": {"rail", "railway", "railways", "railroad", "train", "trains", "track", "tracks", "locomotive",
                     "station", "derailment", "derailed", "intermodal"},
            "road": {"highway", "highways", "truck", "trucks", "trucking", "trucker", "truckers", "traffic",
                     "bridge", "road", "roads", "delivery", "lorry", "hgv", "motorway", "border", "drivers"},
        }

    def assess(self, semantic_score: float, news_context: str, transport_mode: str) -> Dict[str, Any]:
        mode = (transport_mode or "").lower()
        if semantic_score <= 0:
            return {"threat": 0.0, "relevance": 0.0, "rule": "no semantic threat"}
        if mode not in self.relevance_map:
            return {"threat": semantic_score, "relevance": 1.0, "rule": "mode-agnostic leg"}
        tokens = set(re.findall(r"[a-z]+", (news_context or "").lower()))
        own = tokens & self.relevance_map[mode]
        foreign = set().union(*(v for k, v in self.relevance_map.items() if k != mode)) & tokens
        foreign -= self.relevance_map[mode]

        if own:
            relevance, rule = 1.0, f"mode-matched ({', '.join(sorted(own)[:3])})"
        elif tokens & self.AREA_WIDE:
            relevance, rule = 1.0, f"area-wide event ({', '.join(sorted(tokens & self.AREA_WIDE)[:3])})"
        elif foreign:
            relevance, rule = 0.0, f"other-mode event ({', '.join(sorted(foreign)[:3])}) — filtered"
        else:
            relevance, rule = self.UNSPECIFIED_ATTENUATION, "mode unspecified — attenuated"
        return {"threat": round(semantic_score * relevance, 4), "relevance": relevance, "rule": rule}

    def apply_filter(self, semantic_score: float, news_context: str, transport_mode: str) -> float:
        return self.assess(semantic_score, news_context, transport_mode)["threat"]

    def max_pool_threats(self, scores: List[float]) -> float:
        return float(np.max(scores)) if scores else 0.0
