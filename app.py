import os
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Bolzplatzradar",
    page_icon="⚽",
    layout="wide",
)

# Demo-Plätze – später durch echte Daten aus einer Datenbank/API ersetzen.
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


def load_labels():
    """Liest labels.txt aus dem Projektordner."""
    try:
        with open("labels.txt", "r", encoding="utf-8") as file:
            labels = [line.strip() for line in file if line.strip()]
        if labels:
            return labels
    except FileNotFoundError:
        pass

    return ["Leer", "Wenig", "Mittel", "Voll"]


def occupancy_from_players(players):
    """Nur für die Demo-Platzkarten; keine KI-Vorhersage."""
    if players <= 1:
        return "Leer"
    if players <= 3:
        return "Wenig"
    if players <= 5:
        return "Mittel"
    return "Voll"


def occupancy_color(label):
    colors = {
        "Leer": "#17803d",
        "Wenig": "#70a943",
        "Mittel": "#d18b00",
        "Voll": "#c83d3d",
    }
    return colors.get(label, "#667085")


@st.cache_resource
def load_keras_model(model_bytes):
    """Lädt das hochgeladene Modell einmal pro Sitzung/Datei."""
    from tensorflow import keras

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
            temp_file.write(model_bytes)
            temp_path = temp_file.name

        return keras.models.load_model(temp_path, compile=False)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def predict_image(image, model, image_size, labels):
    """
    Macht eine Klassenvorhersage.
    Erwartet RGB-Bilder und Pixelwerte von 0 bis 1.
    Passe die Vorverarbeitung an, falls dein Modell anders trainiert wurde.
    """
    image = image.convert("RGB").resize((image_size, image_size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)

    prediction = np.asarray(model.predict(array, verbose=0)).reshape(-1)
    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])

    if class_index >= len(labels):
        raise ValueError(
            "Das Modell gibt mehr Klassen aus als in labels.txt stehen."
        )

    return labels[class_index], confidence


labels = load_labels()

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

# Seitenleiste: Suche und Filter
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
    st.subheader("🤖 Eigenes Keras-Modell")

    model_file = st.file_uploader(
        "Keras-Modell (.h5)",
        type=["h5"],
        help="Das Modell wird für Vorhersagen verwendet.",
    )

    image_size = st.number_input(
        "Modell-Eingabegröße (Breite und Höhe)",
        min_value=32,
        max_value=1024,
        value=224,
        step=32,
    )

    st.caption(
        "Die Bildgröße, Vorverarbeitung und labels.txt müssen zu deinem "
        "Modelltraining passen."
    )

# Modell laden
model = None
if model_file is not None:
    try:
        model = load_keras_model(model_file.getvalue())
        st.sidebar.success("Modell geladen.")
    except Exception as exc:
        st.sidebar.error(f"Modell konnte nicht geladen werden: {exc}")

# Foto hochladen und klassifizieren
with st.expander("📷 Foto hochladen und Auslastung schätzen"):
    uploaded_photo = st.file_uploader(
        "Bolzplatz-Foto auswählen",
        type=["jpg", "jpeg", "png"],
        key="photo_upload",
    )

    if uploaded_photo is not None:
        try:
            photo = Image.open(uploaded_photo).convert("RGB")
            st.image(photo, caption="Ausgewähltes Foto", use_container_width=True)

            if model is None:
                st.info("Lade zuerst links dein Keras-Modell (.h5) hoch.")
            elif st.button("Auslastung schätzen"):
                predicted_label, confidence = predict_image(
                    photo,
                    model,
                    int(image_size),
                    labels,
                )
                st.success(
                    f"Vorhersage: **{predicted_label}** "
                    f"(Modell-Konfidenz: {confidence:.0%})"
                )
        except Exception as exc:
            st.error(f"Das Foto konnte nicht verarbeitet werden: {exc}")

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
    level = occupancy_from_players(place["players"])

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
        level = occupancy_from_players(place["players"])
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

                st.write("")
                st.write(f"**{place['note']}**")
                st.caption(
                    f"Zuletzt aktualisiert {place['updated']} · Demodaten"
                )
                st.progress(
                    min(place["players"] / 10, 1.0),
                    text="Geschätzte Platzauslastung (Demowert)",
                )
                st.button("Details", key=f"details_{index}")
