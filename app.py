import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Bolzplatzradar",
    page_icon="⚽",
    layout="wide",
)

# Beispielplätze – später durch echte Daten aus einer Datenbank/API ersetzen.
PLACES = [
    {
        "name": "Bolzplatz am Gleisdreieck",
        "area": "Kreuzberg",
        "surface": "Kunstrasen",
        "facilities": "Beleuchtung",
        "players": 8,
        "distance": 0.7,
        "updated": "vor 12 Min.",
        "note": "Gut, wenn du Mitspieler suchst",
        "image": (
            "https://images.unsplash.com/photo-1522778119026-d647f0596c20"
            "?auto=format&fit=crop&w=1000&q=80"
        ),
    },
    {
        "name": "Kreuzberger Kiezplatz",
        "area": "Kreuzberg",
        "surface": "Asphalt",
        "facilities": "Tore",
        "players": 4,
        "distance": 1.2,
        "updated": "vor 25 Min.",
        "note": "Mittlere Auslastung",
        "image": (
            "https://images.unsplash.com/photo-1459865264687-595d652de67e"
            "?auto=format&fit=crop&w=1000&q=80"
        ),
    },
    {
        "name": "Sportplatz im Viktoriapark",
        "area": "Friedrichshain",
        "surface": "Rasen",
        "facilities": "Beleuchtung, Tore",
        "players": 1,
        "distance": 2.1,
        "updated": "vor 8 Min.",
        "note": "Viel freie Fläche",
        "image": (
            "https://images.unsplash.com/photo-1518604666860-9ed391f76460"
            "?auto=format&fit=crop&w=1000&q=80"
        ),
    },
]


def occupancy_from_demo_players(players):
    """Demo-Einteilung anhand der Beispiel-Spielerzahl."""
    if players <= 1:
        return "Leer"
    if players <= 3:
        return "Wenig"
    if players <= 5:
        return "Mittel"
    return "Voll"


def occupancy_color(level):
    return {
        "Leer": "#17803d",
        "Wenig": "#70a943",
        "Mittel": "#d18b00",
        "Voll": "#c83d3d",
    }.get(level, "#667085")


# Kopfbereich
st.markdown(
    """
    <div style="display:flex;align-items:center;gap:12px">
      <span style="font-size:2.4rem">⚽</span>
      <div>
        <h1 style="margin:0">Bolzplatzradar</h1>
        <span style="color:#667085">
          Finde den passenden Bolzplatz – voll für Mitspieler oder frei für Platz.
        </span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Suche und Filter in der Seitenleiste
with st.sidebar:
    st.subheader("📍 Platz suchen")
    search = st.text_input("Ort oder Platzname", placeholder="z. B. Kreuzberg")

    st.subheader("Filter")
    fullness_filter = st.selectbox(
        "Auslastung",
        ["Alle", "Leer", "Wenig", "Mittel", "Voll"],
    )

    surfaces = ["Alle"] + sorted({place["surface"] for place in PLACES})
    surface_filter = st.selectbox("Untergrund", surfaces)

    st.divider()
    st.caption("Die Platzangaben in diesem Prototyp sind Demodaten.")


# Eingebettete Hugging Face Space
st.subheader("📷 Auslastung mit dem Crowd Density Analyzer prüfen")
st.caption(
    "Der Analyzer läuft als separate Hugging Face-App innerhalb dieser Seite."
)

components.iframe(
    "https://peehugahlot05-crowd-density-analyzer.hf.space",
    height=800,
    scrolling=True,
)

st.divider()

# Platzliste
st.subheader("Bolzplätze in deiner Nähe")

sort_option = st.selectbox(
    "Sortieren nach",
    [
        "Empfohlen",
        "Am wenigsten besucht",
        "Am meisten besucht",
        "Entfernung",
    ],
)

filtered_places = []

for place in PLACES:
    level = occupancy_from_demo_players(place["players"])

    if fullness_filter != "Alle" and level != fullness_filter:
        continue

    if surface_filter != "Alle" and place["surface"] != surface_filter:
        continue

    query = search.strip().lower()
    if query and query not in place["name"].lower() and query not in place["area"].lower():
        continue

    filtered_places.append(place)

if sort_option == "Am wenigsten besucht":
    filtered_places.sort(key=lambda place: place["players"])
elif sort_option == "Am meisten besucht":
    filtered_places.sort(key=lambda place: place["players"], reverse=True)
elif sort_option == "Entfernung":
    filtered_places.sort(key=lambda place: place["distance"])

if not filtered_places:
    st.info("Keine Plätze gefunden. Ändere deine Suche oder Filter.")
else:
    for index, place in enumerate(filtered_places):
        level = occupancy_from_demo_players(place["players"])
        color = occupancy_color(level)

        with st.container(border=True):
            image_col, info_col = st.columns([1, 2])

            with image_col:
                st.image(place["image"], use_container_width=True)

            with info_col:
                st.markdown(f"### {place['name']}")
                st.caption(
                    f"📍 {place['area']} · {place['distance']} km entfernt"
                    f" · {place['surface']} · {place['facilities']}"
                )

                st.markdown(
                    f"<span style='background:{color};color:white;"
                    f"padding:5px 10px;border-radius:14px;font-weight:600'>"
                    f"{level} · {place['players']} Spieler</span>",
                    unsafe_allow_html=True,
                )

                st.write(f"**{place['note']}**")
                st.caption(
                    f"Zuletzt aktualisiert {place['updated']} · Demodaten"
                )
                st.progress(
                    min(place["players"] / 10, 1.0),
                    text="Beispiel-Auslastung",
                )
                st.button("Details", key=f"details_{index}")
