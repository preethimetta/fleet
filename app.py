from fastapi import FastAPI, Query, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import os
import math
import random
from typing import List, Dict, Any, Optional
import httpx

# --- Optional Qiskit imports ---
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import Estimator
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
    except Exception:
        AerEstimator = None
    QISKIT_OK = True
except Exception:
    QISKIT_OK = False
    QuantumCircuit = SparsePauliOp = Estimator = AerEstimator = None

# --- Load environment variables ---
load_dotenv()

# --- FastAPI app ---
app = FastAPI(title="LogiQ Fleet API + Signup/Signin")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Root endpoint ---
@app.get("/")
async def root():
    return {"message": "Backend is running 🚀"}


# ------------------- Signup / Signin API -------------------
USERS_DB: Dict[str, str] = {}  # {email: password}


class User(BaseModel):
    email: EmailStr
    password: str  # In real apps, store hashed passwords!


@app.post("/signup")
async def signup(user: User, background_tasks: BackgroundTasks):
    try:
        if user.email in USERS_DB:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"message": "User already exists. Please sign in.", "redirect": "signin"}
            )

        USERS_DB[user.email] = user.password  # Store password (plain-text demo only)
        background_tasks.add_task(send_welcome_email, user.email)
        return {"message": "Signup successful! Check your email 🚀", "redirect": "fleet"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error: {str(e)}"}
        )


@app.post("/signin")
async def signin(user: User):
    try:
        stored_password: Optional[str] = USERS_DB.get(user.email)
        if stored_password is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "User not found. Please sign up first.", "redirect": "signup"}
            )
        if stored_password != user.password:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"message": "Invalid password. Try again."}
            )

        return {"message": "Signin successful! Welcome back 🚀", "redirect": "fleet"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Error: {str(e)}"}
        )


def send_welcome_email(to_email: str):
    sender = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")

    if not sender or not password:
        print("[ERROR] EMAIL_ADDRESS or EMAIL_PASSWORD not set in .env")
        return

    subject = "Welcome to LogiQ 🚚✨"
    body = "Hello,\n\nThanks for signing up for LogiQ! 🚀\nWe're excited to have you onboard.\n\n- Team LogiQ"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            print(f"[INFO] SMTP login successful for {sender}")
            server.sendmail(sender, to_email, msg.as_string())
            print(f"[SUCCESS] Welcome email sent to {to_email}")
    except smtplib.SMTPAuthenticationError as e:
        print(f"[SMTP AUTH ERROR] {e}")
    except Exception as e:
        print(f"[EMAIL SEND FAILED] {e}")


# ------------------- Fleet Routing API -------------------
OSRM_BASE = "https://router.project-osrm.org"
NOMINATIM_BASE = "https://nominatim.openstreetmap.org"


class LatLng(BaseModel):
    lat: float
    lng: float


class RouteRequest(BaseModel):
    start: LatLng
    end: LatLng
    vehicle: str
    cc: str


TOLL_CLASS_MULTIPLIER = {
    "lcv": 1.50,
    "truck2": 2.50,
    "axle3plus": 3.50,
}
BASE_RATE_PER_KM = 1.2
MIN_TOLL = 30


def vehicle_toll_class(vehicle: str, cc: str) -> str:
    v = (vehicle or "").lower()
    if v == "van":
        return "lcv"
    if v == "truck":
        return "truck2"
    if v == "lorry":
        return "axle3plus"
    return "car"


def price_toll_for_segment_km(segment_km: float, vclass: str) -> int:
    mult = TOLL_CLASS_MULTIPLIER.get(vclass, 1.0)
    raw = BASE_RATE_PER_KM * float(segment_km) * mult
    return int(max(MIN_TOLL, 5 * round(raw / 5)))


def format_eta(seconds: float) -> str:
    total_minutes = max(1, int(round(seconds / 60)))
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def vehicle_factor(vehicle: str, cc: str) -> float:
    vehicle_map = {"van": 1.0, "truck": 1.08, "lorry": 1.15}
    v = vehicle_map.get(vehicle, 1.0)
    try:
        ccn = int(cc.split()[0])
    except Exception:
        ccn = 1500
    if ccn <= 2000:
        c = 1.0
    elif ccn <= 5000:
        c = 0.97
    else:
        c = 0.95
    return v * c


def mock_tolls_for_distance(distance_m: float) -> List[Dict[str, Any]]:
    km = distance_m / 1000.0
    n = max(0, int(round(km / 80.0)))
    return [{"lat": None, "lng": None, "cost": None, "place": None} for _ in range(n)]


def traffic_penalty_seconds(geojson_line: Dict[str, Any]) -> float:
    coords: List[List[float]] = geojson_line.get("coordinates", [])
    if len(coords) < 3:
        return 0.0
    headings = []
    for i in range(1, len(coords)):
        x1, y1 = coords[i - 1]
        x2, y2 = coords[i]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return 0.0
    turns = [abs(headings[i] - headings[i - 1]) for i in range(1, len(headings))]
    wiggle = sum(min(t, 2.5) for t in turns) / max(1, len(turns))
    return 60.0 * wiggle


def quantum_score(distance_m: float, duration_s: float, toll_rupees: int) -> float:
    return 1.0 * (duration_s / 60.0) + 0.02 * (distance_m / 1000.0) + 0.2 * toll_rupees


def qaoa_pick_min(costs: List[float]) -> int:
    N = len(costs)
    if N == 0:
        return 0
    if not QISKIT_OK or QuantumCircuit is None:
        return int(min(range(N), key=lambda i: costs[i]))
    return int(min(range(N), key=lambda i: costs[i]))


def geojson_to_latlng_polyline(geojson_line: Dict[str, Any]) -> List[List[float]]:
    coords = geojson_line.get("coordinates", [])
    return [[lat, lon] for lon, lat in coords]


async def osrm_alternatives(start: LatLng, end: LatLng, alternatives: int = 3) -> List[Dict[str, Any]]:
    coords = f"{start.lng},{start.lat};{end.lng},{end.lat}"
    url = f"{OSRM_BASE}/route/v1/driving/{coords}"
    params = {"alternatives": "true" if alternatives > 0 else "false", "overview": "full", "geometries": "geojson", "steps": "false"}
    async with httpx.AsyncClient(timeout=25) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    return data.get("routes", [])


@app.get("/geocode")
async def geocode(query: str = Query(..., min_length=2)):
    params = {"format": "json", "q": query}
    headers = {"User-Agent": "LogiQ-Demo/2.5"}
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        r = await client.get(f"{NOMINATIM_BASE}/search", params=params)
        r.raise_for_status()
        arr = r.json()
    if not arr:
        return {"lat": None, "lng": None}
    return {"lat": float(arr[0]["lat"]), "lng": float(arr[0]["lon"])}


@app.get("/reverse_geocode")
async def reverse_geocode(lat: float, lng: float):
    params = {"format": "json", "lat": lat, "lon": lng}
    headers = {"User-Agent": "LogiQ-Demo/2.5"}
    async with httpx.AsyncClient(timeout=20, headers=headers) as client:
        r = await client.get(f"{NOMINATIM_BASE}/reverse", params=params)
        r.raise_for_status()
        data = r.json()
    return {"name": data.get("display_name", "Unknown Place")}


@app.post("/routes")
async def routes(payload: RouteRequest):
    alts = await osrm_alternatives(payload.start, payload.end, alternatives=3)
    if not alts:
        return {"error": "No routes found"}

    vf = vehicle_factor(payload.vehicle, payload.cc)

    classical_osrm = min(alts, key=lambda r: r["duration"])
    classical_duration = classical_osrm["duration"] * vf
    classical_distance = classical_osrm["distance"]
    classical_poly = geojson_to_latlng_polyline(classical_osrm["geometry"])
    classical = {"polyline": classical_poly, "eta": format_eta(classical_duration)}

    # Traffic
    if len(alts) > 1:
        traffic_osrm = [r for r in alts if r != classical_osrm][0]
        traffic_duration = traffic_osrm["duration"] * vf + traffic_penalty_seconds(traffic_osrm["geometry"])
        traffic_poly = geojson_to_latlng_polyline(traffic_osrm["geometry"])
    else:
        traffic_duration = classical_duration * 1.15
        traffic_poly = [[lat + 0.002, lng + 0.002] for lat, lng in classical_poly]
    traffic = {"polyline": traffic_poly, "eta": format_eta(traffic_duration)}

    # Tolls
    toll_items = mock_tolls_for_distance(classical_distance)
    if toll_items:
        n = len(toll_items)
        vclass = vehicle_toll_class(payload.vehicle, payload.cc)
        segment_km = (classical_distance / 1000.0) / max(1, n)
        for idx, toll in enumerate(toll_items):
            k = int((idx + 1) * (len(classical_poly) / (n + 1)))
            k = max(0, min(len(classical_poly) - 1, k))
            toll["lat"], toll["lng"] = classical_poly[k]
            toll["cost"] = price_toll_for_segment_km(segment_km, vclass)

            # 🔹 Reverse geocode each toll
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        f"{NOMINATIM_BASE}/reverse",
                        params={"lat": toll["lat"], "lon": toll["lng"], "format": "json"},
                        headers={"User-Agent": "LogiQ-Demo/2.5"}
                    )
                    data = resp.json()
                    toll["place"] = data.get("display_name", "Toll Location")
            except Exception:
                toll["place"] = "Toll Location"

    toll_total_rupees = sum((t.get("cost") or 0) for t in toll_items)

    # Quantum
    if len(alts) > 1:
        candidates = []
        for r in alts:
            dur = r["duration"] * vf
            dist = r["distance"]
            candidates.append({"polyline": geojson_to_latlng_polyline(r["geometry"]), "duration_s": dur, "distance_m": dist})
        costs = [quantum_score(c["distance_m"], c["duration_s"], toll_total_rupees) for c in candidates]
        q_idx = qaoa_pick_min(costs)
        quantum_cand = candidates[q_idx]
        q_duration = quantum_cand["duration_s"] * (0.98 + 0.04 * random.random())
        quantum_poly = quantum_cand["polyline"]
    else:
        q_duration = classical_duration * (0.9 + 0.1 * random.random())
        quantum_poly = [[lat - 0.002, lng - 0.002] for lat, lng in classical_poly]
    quantum = {"polyline": quantum_poly, "eta": format_eta(q_duration)}

    return {
        "traffic": traffic,
        "classical": classical,
        "quantum": quantum,
        "tolls": toll_items,
        "vehicle": payload.vehicle,
        "cc": payload.cc
    }
