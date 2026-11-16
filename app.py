import os
import tempfile

import streamlit as st
from gradio_client import Client, handle_file
from PIL import Image

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


@st.cache_resource
def get_huggingface_client():
    """Verbindet sich mit dem Crowd Density Analyzer auf Hugging Face."""
    return Client("peehugahlot05/crowd-density-analyzer")


def occupancy_from_demo_players(players):
    """Demo-Einteilung für die Beispielplätze."""
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


def show_api_image(image_result, caption):
    """Zeigt ein Bild an, das die Gradio-API zurückgegeben hat."""
    if not image_result:
        st.info(f"{caption}: kein Bild zurückgegeben.")
        return

    if isinstance(image_result, dict):
        image_path = image_result.get("path")
        image_url = image_result.get("url")

        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=caption, use_container_width=True)
        elif image_url:
            st.image(image_url, caption=caption, use_container_width=True)
        else:
            st.info(f"{caption}: Bilddatei nicht verfügbar.")
    elif isinstance(image_result, str):
        if os.path.exists(image_result):
            st.image(image_result, caption=caption, use_container_width=True)
        else:
            st.image(image_result, caption=caption, use_container_width=True)
    else:
        st.info(f"{caption}: unbekanntes Bildformat.")


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

# Suche und Filter
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


# Fotoanalyse: Das Foto wird an den Hugging Face Space gesendet.
st.subheader("📷 Bolzplatzfoto analysieren")
st.write(
    "Lade ein Foto hoch. Der Crowd Density Analyzer erkennt Personen "
    "und zeigt seine Analysebilder hier an."
)

uploaded_photo = st.file_uploader(
    "Platzfoto auswählen",
    type=["jpg", "jpeg", "png"],
    key="place_photo",
)

if uploaded_photo is not None:
    try:
        photo = Image.open(uploaded_photo).convert("RGB")
        st.image(
            photo,
            caption="Ausgewähltes Foto",
            width=500,
        )

        if st.button("Foto analysieren", type="primary"):
            try:
                client = get_huggingface_client()

                # Gradio benötigt für den Upload einen Dateipfad.
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    delete=False,
                ) as temp_file:
                    photo.save(temp_file.name, format="JPEG")
                    temp_path = temp_file.name

                try:
                    with st.spinner(
                        "Das Foto wird vom Crowd Density Analyzer ausgewertet..."
                    ):
                        result = client.predict(
                            image=handle_file(temp_path),
                            api_name="/analyze",
                        )
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

                if not isinstance(result, (list, tuple)) or len(result) < 4:
                    st.error("Der Analyzer hat ein unerwartetes Ergebnis geliefert.")
                else:
                    st.subheader("Analyseergebnis")

                    # API-Ausgabe 0: HTML-Auswertung
                    if result[0]:
                        st.markdown(
                            str(result[0]),
                            unsafe_allow_html=True,
                        )

                    # API-Ausgabe 2: YOLO-Erkennungsbild
                    # API-Ausgabe 3: CSRNet-Heatmap
                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        show_api_image(result[2], "YOLO Detection")

                    with result_col2:
                        show_api_image(result[3], "CSRNet Heatmap")

            except Exception as exc:
                st.error(f"Die Analyse ist fehlgeschlagen: {exc}")

    except Exception as exc:
        st.error(f"Das Foto konnte nicht geöffnet werden: {exc}")


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
