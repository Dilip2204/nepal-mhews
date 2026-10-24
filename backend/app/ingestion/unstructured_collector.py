"""Unstructured Field Incident Bulletins & Satellite Visual Candidate Collector.

Collects and stages raw disaster observations from public feeds, news RSS,
and satellite anomaly detections into the database queue (PostGIS or SQLite)
WITHOUT calling external LLM APIs during daily cycles.

Staged data is subsequently audited and parsed by the periodic / weekly AI audit engine
(`backend/scripts/run_weekly_ai_audit.py`).
"""

import hashlib
import json
import logging
import math
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.database import get_time_series_store, BaseTimeSeriesStore

logger = logging.getLogger("nepal_mhews.unstructured_collector")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

# 77 Districts of Nepal for spatial keyword tagging
NEPAL_DISTRICTS = [
    "Achham", "Arghakhanchi", "Baglung", "Baitadi", "Bajhang", "Bajura", "Banke", "Bardiya",
    "Bhaktapur", "Bhojpur", "Chitwan", "Dadeldhura", "Dailekh", "Dang", "Darchula", "Dhading",
    "Dhankuta", "Dhanusha", "Dolakha", "Dolpa", "Doti", "Gorkha", "Gulmi", "Humla",
    "Ilam", "Jajarkot", "Jhapa", "Jumla", "Kailali", "Kalikot", "Kanchanpur", "Kapilvastu",
    "Kaski", "Kavrepalanchok", "Khotang", "Lalitpur", "Lamjung", "Mahottari", "Makwanpur", "Manang",
    "Morang", "Mugu", "Mustang", "Myagdi", "Nawalpur", "Nuwakot", "Okhaldhunga", "Palpa",
    "Panchthar", "Parbat", "Parsa", "Parasi", "Pyuthan", "Ramechhap", "Rasuwa", "Rautahat",
    "Rolpa", "Rukum", "Rupandehi", "Salyan", "Sankhuwasabha", "Saptari", "Sarlahi", "Sindhuli",
    "Sindhupalchok", "Siraha", "Solukhumbu", "Sunsari", "Surkhet", "Syangja", "Tanahun", "Taplejung",
    "Terhathum", "Udayapur"
]

DISASTER_KEYWORDS = [
    "landslide", "landslides", "mudslide", "debris flow", "rockfall", "rock-ice", "avalanche",
    "slope failure", "highway blocked", "glacier", "glof", "burst", "cloudburst", "flooding",
    "पहिरो", "पहिरोले", "पहिरोका", "पहिराले", "हिमपहिरो", "बाढी", "सडक अवरुद्ध"
]

RSS_FEED_SOURCES = [
    {
        "source": "GOOGLE_NEWS_EN",
        "url": "https://news.google.com/rss/search?q=landslide+Nepal+when:7d&hl=en-US&gl=US&ceid=US:en",
    },
    {
        "source": "GOOGLE_NEWS_NEPALI",
        "url": "https://news.google.com/rss/search?q=%E0%A4%AA%E0%A4%B9%E0%A4%BF%E0%A4%B0%E0%A5%8B+when:7d&hl=ne&gl=NP&ceid=NP:ne",
    },
    {
        "source": "KATHMANDU_POST_RSS",
        "url": "https://kathmandupost.com/rss",
    },
]

# Curated reference disaster bulletins for high-fidelity offline demonstration and seeding
SEED_DISASTER_BULLETINS: List[Dict[str, Any]] = [
    {
        "source": "NDRRMA_OFFICIAL_BULLETIN",
        "title": "Jhyaple Khola Prithvi Highway Mass Debris Flow Disaster",
        "raw_text": (
            "National Disaster Risk Reduction & Management Authority (NDRRMA) Incident Update: "
            "Continuous torrential cloudburst rainfall (>320mm in 48 hours) caused a catastrophic debris flow "
            "at Jhyaple Khola, Dhading district along the Tribhuvan/Prithvi highway corridor (approx coordinates 27.712N, 85.218E). "
            "Slopes above 38 degrees failed rapidly, burying multiple commuter vehicles in mud and boulders. "
            "Rescue teams deployed. Roadway blocked for three days."
        ),
        "url": "https://bipadportal.gov.np/incidents/jhyaple_khola_2024",
        "district_hint": "Dhading",
    },
    {
        "source": "NEPAL_POLICE_DISASTER_DISPATCH",
        "title": "Bhotekoshi Araniko Highway Rockfall near Tatopani",
        "raw_text": (
            "सिन्धुपाल्चोक जिल्लाको भोटेकोशी गाउँपालिका तातोपानी नजिकै अरनिको राजमार्गमा ठूलो सुक्खा पहिरो गएको छ। "
            "पहिराले गर्दा चीनसँगको सीमा जोड्ने सडक पूर्ण रूपमा अवरुद्ध भएको छ। "
            "ठाडो भिर (ढलान ४२ डिग्री) बाट ढुङ्गा माटो खसेको छ। करिब २७.९६८ उत्तर अक्षांश र ८५.९४२ पूर्व देशान्तर नजिकै।"
        ),
        "url": "https://nepalpolice.gov.np/updates/araniko-bhotekoshi-landslide",
        "district_hint": "Sindhupalchok",
    },
    {
        "source": "HIMALAYAN_CRYOSPHERE_EXPEDITION",
        "title": "Thame Khumbu Upper Moraine Dam Breach and Outburst",
        "raw_text": (
            "Field Glaciological Report: In Solukhumbu district, Khumbu Pasanglhamu Rural Municipality above Thame village "
            "(elevation ~4780m, coordinates 27.831N, 86.612E), twin glacial moraine dams breached following weeks of sub-surface permafrost thaw. "
            "A sudden surge of meltwater and rock-ice debris scoured the valley floor, washing away downstream bridges and schools. "
            "Sentinel-1 SAR amplitude diff confirms 350m headwall scarp."
        ),
        "url": "https://dhm.gov.np/cryosphere/thame_2024_report",
        "district_hint": "Solukhumbu",
    },
    {
        "source": "KATHMANDU_POST_ARCHIVE",
        "title": "Manaslu Birendra Tal Glacier Icefall Surge (Gorkha)",
        "raw_text": (
            "A massive hanging serac collapsed from Mt Manaslu into Birendra Tal in Gorkha district (28.596N, 84.636E, elevation 3620m). "
            "The displacement wave overtopped the terminal moraine and sent a debris-laden flash flood down the Budhi Gandaki river gorge "
            "towards Samagaun. Wooden bridges destroyed; no human casualties due to upstream warning."
        ),
        "url": "https://kathmandupost.com/national/2024/04/21/manaslu-birendra-tal-overflow",
        "district_hint": "Gorkha",
    },
    {
        "source": "ROSHI_WATERSHED_COMMUNITY_ALERT",
        "title": "Roshi Khola BP Highway Landslide Choke (Kavre)",
        "raw_text": (
            "काभ्रेपलाञ्चोक जिल्लाको रोशी खोला किनारस्थित बिपी राजमार्गमा मुसलधारे वर्षाका कारण पहिरो र लेदो बगेको छ। "
            "डाँडाकटहरे नजिकै २७.५४८ उत्तर र ८५.६४२ पूर्वमा नदी थुनिएर ताल बनेको छ। "
            "उच्च सतर्कता अपनाउन स्थानीय प्रशासनले अनुरोध गरेको छ।"
        ),
        "url": "https://kavrepost.com/roshi-bp-highway-flood-2024",
        "district_hint": "Kavrepalanchok",
    },
]


NEPAL_DISTRICTS_DEVANAGARI: Dict[str, str] = {
    "गोरखा": "Gorkha",
    "रसुवा": "Rasuwa",
    "सिन्धुपाल्चोक": "Sindhupalchok",
    "धादिङ": "Dhading",
    "काभ्रेपलाञ्चोक": "Kavrepalanchok",
    "काभ्रे": "Kavrepalanchok",
    "दोलखा": "Dolakha",
    "सोलुखुम्बु": "Solukhumbu",
    "कास्की": "Kaski",
    "म्याग्दी": "Myagdi",
    "मनाङ": "Manang",
    "मुस्ताङ": "Mustang",
    "लमजुङ": "Lamjung",
    "संखुवासभा": "Sankhuwasabha",
    "ताप्लेजुङ": "Taplejung",
    "चितवन": "Chitwan",
    "मकवानपुर": "Makwanpur",
    "काठमाडौं": "Kathmandu",
    "ललितपुर": "Lalitpur",
    "भक्तपुर": "Bhaktapur",
    "नुवाकोट": "Nuwakot",
    "बाजुरा": "Bajura",
    "बझाङ": "Bajhang",
    "दार्चुला": "Darchula",
    "जुम्ला": "Jumla",
    "हुम्ला": "Humla",
    "कालिकोट": "Kalikot",
    "मुगु": "Mugu",
    "डोल्पा": "Dolpa",
    "रोल्पा": "Rolpa",
    "रुकुम": "Rukum",
    "जाजरकोट": "Jajarkot",
    "पर्वत": "Parbat",
    "बागलुङ": "Baglung",
    "स्याङ्जा": "Syangja",
    "तनहुँ": "Tanahun",
    "गुल्मी": "Gulmi",
    "पाल्पा": "Palpa",
    "अर्घाखाँची": "Arghakhanchi",
}


def extract_district_hint(text: str) -> Optional[str]:
    """Scan text for mentions of any of Nepal's 77 districts in English or Nepali Devanagari."""
    text_lower = text.lower()
    for d in NEPAL_DISTRICTS:
        if d.lower() in text_lower:
            return d
    for dev_name, eng_name in NEPAL_DISTRICTS_DEVANAGARI.items():
        if dev_name in text:
            return eng_name
    return None


def is_geohazard_relevant(title: str, text: str) -> bool:
    """Check if the text contains geohazard or disaster related keywords."""
    combined = (title + " " + text).lower()
    return any(kw in combined for kw in DISASTER_KEYWORDS)


def fetch_open_bulletins_and_rss(timeout_s: int = 8) -> List[Dict[str, Any]]:
    """Fetch recent hazard bulletins and news articles from open RSS channels."""
    collected: List[Dict[str, Any]] = []

    for src in RSS_FEED_SOURCES:
        try:
            req = urllib.request.Request(
                src["url"],
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Nepal-MHEWS/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                content = response.read()
                root = ET.fromstring(content)
                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    desc_elem = item.find("description")

                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                    desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

                    # Clean HTML tags from description if present
                    clean_desc = re.sub(r"<[^>]+>", " ", desc).strip()

                    if title and is_geohazard_relevant(title, clean_desc):
                        dist = extract_district_hint(title + " " + clean_desc)
                        collected.append({
                            "source": src["source"],
                            "title": title,
                            "raw_text": f"{title}. {clean_desc}",
                            "url": link,
                            "district_hint": dist,
                            "received_utc": time.time(),
                        })
        except Exception as e:
            logger.debug("RSS fetch failed for %s (%s)", src["source"], e)

    return collected


def stage_unprocessed_bulletins(store: Optional[BaseTimeSeriesStore] = None) -> int:
    """Ingest seed bulletins (if enabled) and live RSS items into the `unprocessed_bulletins` queue."""
    from app.core.config import settings

    if store is None:
        store = get_time_series_store()

    items = []
    if getattr(settings, "seed_demo_bulletins_enabled", False):
        items.extend(SEED_DISASTER_BULLETINS)
    live_items = fetch_open_bulletins_and_rss()
    items.extend(live_items)

    count_staged = 0
    for it in items:
        try:
            bid_hash = hashlib.md5((it["title"] + it["source"]).encode("utf-8")).hexdigest()[:12]
            it_record = {
                "bulletin_id": f"bul_{bid_hash}",
                "received_utc": it.get("received_utc", time.time()),
                "source": it["source"],
                "title": it["title"],
                "raw_text": it["raw_text"],
                "url": it.get("url"),
                "district_hint": it.get("district_hint"),
                "status": "PENDING",
            }
            store.insert_unprocessed_bulletin(it_record)
            count_staged += 1
        except Exception as e:
            logger.debug("Failed staging bulletin '%s': %s", it.get("title"), e)

    return count_staged


def generate_synthetic_thumbnail_pair(
    site_id: str,
    lat: float,
    lon: float,
    is_true_scar: bool = True,
    size_px: int = 512,
) -> Tuple[Path, Path]:
    """Generate a realistic 512x512 before/after thumbnail pair for multimodal vision screening.
    
    If `is_true_scar=True`:
      - 'Before': Lush green mountain slope (NDVI ~0.72)
      - 'After': Bare earth / exposed rock rupture scarp with elongated downslope debris runout trail (NDVI ~0.24)
    If `is_true_scar=False`:
      - 'Before': Green/alpine slope
      - 'After': Fluffy cloud cover with diffuse grey cast shadow or snowfield boundary (false alarm)
    """
    from PIL import Image, ImageDraw

    before_path = THUMBNAILS_DIR / f"{site_id}_before.png"
    after_path = THUMBNAILS_DIR / f"{site_id}_after.png"

    # Base mountain terrain (green gradient with relief shading)
    img_before = Image.new("RGB", (size_px, size_px), color=(54, 98, 48))
    draw_before = ImageDraw.Draw(img_before)
    for y in range(size_px):
        # Topographic contour ridges
        shade = int(20.0 * math.sin(y / 18.0) + 15.0 * math.cos(y / 35.0))
        draw_before.line([(0, y), (size_px, y)], fill=(54 + shade, 98 + shade, 48 + shade // 2))

    # Add mountain ridge line and vegetation stippling
    draw_before.polygon([(0, 0), (size_px, 0), (size_px, 120), (0, 180)], fill=(85, 120, 68))
    img_before.save(before_path)

    img_after = img_before.copy()
    draw_after = ImageDraw.Draw(img_after)

    if is_true_scar:
        # Paint exposed rocky scarp (crown) and elongated downslope debris chute
        # Earth tones: brown, ochre, grey rubble
        scarp_crown = [(210, 160), (310, 150), (330, 210), (280, 260), (190, 220)]
        debris_track = [
            (210, 200), (310, 190), (340, 310), (310, 420),
            (280, 500), (220, 490), (230, 380), (180, 280)
        ]
        draw_after.polygon(debris_track, fill=(156, 128, 92))
        draw_after.polygon(scarp_crown, fill=(128, 98, 70))
        # Highlight scarp rupture rim
        draw_after.line([(210, 160), (310, 150)], fill=(210, 190, 160), width=3)
    else:
        # Cloud shadow or snow patch (diffuse semi-transparent polygon)
        # Cloud white mass
        draw_after.ellipse([(140, 80), (380, 240)], fill=(245, 248, 252))
        # Cast shadow (darkened area shifted south-east)
        shadow_box = [(170, 260), (410, 260), (390, 380), (150, 380)]
        draw_after.polygon(shadow_box, fill=(28, 48, 25))

    img_after.save(after_path)
    return before_path, after_path


def stage_satellite_visual_candidates(store: Optional[BaseTimeSeriesStore] = None) -> int:
    """Generate and queue high-priority visual inspection candidates for the weekly audit."""
    if store is None:
        store = get_time_series_store()

    # Pre-configure 4 benchmark visual candidates representing diverse geomorphic situations:
    # 2 true scars (Tatopani & Jhyaple Khola) and 2 false artifacts (cloud shadow & snowline shift)
    candidate_specs = [
        {
            "site_id": "vis_cand_tatopani_scarp",
            "lat": 27.968,
            "lon": 85.942,
            "trigger": "OPTICAL_DELTA_NDVI",
            "delta_ndvi": -0.28,
            "sar_vel": 18.5,
            "is_true_scar": True,
            "notes": "Sentinel-2 detected 28% canopy drop along Araniko Highway gorge.",
        },
        {
            "site_id": "vis_cand_jhyaple_flow",
            "lat": 27.712,
            "lon": 85.218,
            "trigger": "EXTREME_TRIGGER_CRITICAL",
            "delta_ndvi": -0.32,
            "sar_vel": 24.0,
            "is_true_scar": True,
            "notes": "Mass debris chute following 312mm rainfall cloudburst.",
        },
        {
            "site_id": "vis_cand_langtang_cloud_shadow",
            "lat": 28.215,
            "lon": 85.515,
            "trigger": "OPTICAL_DELTA_NDVI",
            "delta_ndvi": -0.21,
            "sar_vel": 1.2,
            "is_true_scar": False,
            "notes": "Candidate anomaly near Kyangjin Gompa; suspected monsoon cumulus cast shadow.",
        },
        {
            "site_id": "vis_cand_manaslu_snowmelt_boundary",
            "lat": 28.540,
            "lon": 84.630,
            "trigger": "OPTICAL_DELTA_NDVI",
            "delta_ndvi": -0.19,
            "sar_vel": 2.1,
            "is_true_scar": False,
            "notes": "High elevation (4400m) reflectance shift; suspected seasonal nival snowline retreat.",
        },
    ]

    count_staged = 0
    for spec in candidate_specs:
        before_p, after_p = generate_synthetic_thumbnail_pair(
            site_id=spec["site_id"],
            lat=spec["lat"],
            lon=spec["lon"],
            is_true_scar=spec["is_true_scar"],
        )
        rec = {
            "candidate_id": spec["site_id"],
            "detected_utc": time.time(),
            "site_id": spec["site_id"],
            "latitude": spec["lat"],
            "longitude": spec["lon"],
            "trigger_type": spec["trigger"],
            "delta_ndvi": spec["delta_ndvi"],
            "sar_velocity_mm_month": spec["sar_vel"],
            "thumbnail_before_path": str(before_p),
            "thumbnail_after_path": str(after_p),
            "status": "PENDING_AI_INSPECTION",
            "ai_confidence": None,
            "ai_notes": spec["notes"],
        }
        store.insert_visual_candidate(rec)
        count_staged += 1

    return count_staged


if __name__ == "__main__":
    store = get_time_series_store()
    print("Staging unstructured disaster news & bulletins...")
    b_count = stage_unprocessed_bulletins(store)
    print(f"Staged {b_count} bulletin items into database.")

    print("Staging satellite visual candidates & generating thumbnails...")
    v_count = stage_satellite_visual_candidates(store)
    print(f"Staged {v_count} visual candidates into database.")
