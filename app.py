from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Bolzplatzradar",
    page_icon="⚽",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "bolzplatz_model.h5"
LABELS_PATH = BASE_DIR / "labels.txt"
IMAGE_SIZE = 224  # Muss zur Eingabegröße deines Modells passen.

# Beispielplätze: später durch echte Daten aus einer Datenbank/API ersetzen.
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
    """Lädt die Klassen in der Reihenfolge aus labels.txt."""
    if not LABELS_PATH.exists():
        return ["Leer", "Wenig", "Mittel", "Voll"]

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file if line.strip()]

    return labels or ["Leer", "Wenig", "Mittel", "Voll"]


@st.cache_resource
def load_model():
    """Lädt das fest im Projekt hinterlegte Keras-Modell."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_occupancy(image, model, labels):
    """
    Klassifiziert ein Foto.

    Passe Größe und Normalisierung an, falls dein Modell beim Training
    eine andere Bildvorverarbeitung verwendet hat.
    """
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = np.asarray(
        model.predict(image_array, verbose=0)
    ).reshape(-1)

    if len(prediction) != len(labels):
        raise ValueError(
            f"Das Modell gibt {len(prediction)} Werte aus, "
            f"aber labels.txt enthält {len(labels)} Klassen."
        )

    class_index = int(np.argmax(prediction))
    confidence = float(prediction[class_index])

    return labels[class_index], confidence


def occupancy_from_demo_players(players):
    """Demo-Einteilung für die Beispielkarten."""
    if players <= 1:
        return "Leer"
    if players <= 3:
        return "Wenig"
    if players <= 5:
        return "Mittel"
    return "Voll"


def occupancy_color(label):
    return {
        "Leer": "#17803d",
        "Wenig": "#70a943",
        "Mittel": "#d18b00",
        "Voll": "#c83d3d",
    }.get(label, "#667085")


# Modell beim Start laden. Es gibt keinen Modell-Upload in der Benutzeroberfläche.
try:
    model = load_model()
    model_error = None
except Exception as exc:
    model = None
    model_error = str(exc)


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

if model is None:
    st.warning(
        "Das KI-Modell ist gerade nicht verfügbar. "
        "Prüfe, ob models/bolzplatz_model.h5 im Projekt liegt."
    )
    with st.expander("Technische Information"):
        st.code(model_error or "Unbekannter Fehler")


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


# Foto-Upload und Vorhersage
with st.expander("📷 Foto eines Bolzplatzes hochladen"):
    st.caption(
        "Das Foto wird mit dem im Projekt hinterlegten Modell ausgewertet. "
        "Es wird durch diesen Demo-Code nicht dauerhaft gespeichert."
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
                caption="Ausgewähltes Platzfoto",
                use_container_width=True,
            )

            if model is not None:
                if st.button("Auslastung schätzen"):
                    labels = load_labels()

                    try:
                        prediction_label, confidence = predict_occupancy(
                            photo,
                            model,
                            labels,
                        )
                        st.success(
                            f"Geschätzte Auslastung: **{prediction_label}** "
                            f"(Modell-Konfidenz: {confidence:.0%})"
                        )
                    except Exception as exc:
                        st.error(f"Die Vorhersage ist fehlgeschlagen: {exc}")
            else:
                st.info("Die Schätzung ist nicht verfügbar, solange das Modell fehlt.")

        except Exception as exc:
            st.error(f"Das Foto konnte nicht geöffnet werden: {exc}")


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
