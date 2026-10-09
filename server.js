
"use strict";

// =====================================================
// MeineEigeneKI – server.js
// Eigener HTTP-Server ohne externe Bibliotheken
//
// Projekt:
//   server.js
//   netz.js
//   index.html
//   Daten/training.json
// =====================================================

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { NeuronalesNetz } = require("./netz.js");

const PORT = Number(process.env.PORT) || 3000;
const HOST = process.env.HOST || "127.0.0.1";

const BASE_DIR = __dirname;
const HTML_FILE = path.join(BASE_DIR, "index.html");
const DATA_FILE = path.join(BASE_DIR, "Daten", "training.json");

const TRAINING_EPOCHS = 20000;
const MAX_BODY_SIZE = 10000;

// =====================================================
// TRAININGSDATEN LADEN
// =====================================================

function ladeTrainingsdaten() {
  if (!fs.existsSync(DATA_FILE)) {
    throw new Error(
      "Trainingsdatei nicht gefunden: " + DATA_FILE
    );
  }

  let daten;

  try {
    daten = JSON.parse(fs.readFileSync(DATA_FILE, "utf8"));
  } catch (error) {
    throw new Error(
      "Daten/training.json enthält ungültiges JSON: " +
      error.message
    );
  }

  if (!Array.isArray(daten) || daten.length === 0) {
    throw new Error(
      "Die Trainingsdatei muss ein nicht leeres Array enthalten."
    );
  }

  for (let i = 0; i < daten.length; i++) {
    const eintrag = daten[i];

    if (
      !eintrag ||
      ![0, 1].includes(eintrag.x1) ||
      ![0, 1].includes(eintrag.x2) ||
      ![0, 1].includes(eintrag.target)
    ) {
      throw new Error(
        "Ungültiger Eintrag in training.json, Position " + i +
        ". Erforderlich sind x1, x2 und target mit jeweils 0 oder 1."
      );
    }
  }

  console.log("Trainingsbeispiele geladen:", daten.length);
  return daten;
}

// =====================================================
// NEURONALES NETZ INITIALISIEREN UND TRAINIEREN
// =====================================================

const trainingsdaten = ladeTrainingsdaten();
const netz = new NeuronalesNetz();

console.log("Trainiere das neuronale Netz ...");

const trainingsfehler = netz.trainiere(
  trainingsdaten,
  TRAINING_EPOCHS
);

console.log("Training abgeschlossen.");
console.log("Trainingsdurchläufe:", TRAINING_EPOCHS);
console.log("Mittlerer quadratischer Fehler:", trainingsfehler);

// =====================================================
// JSON-ANTWORTEN
// =====================================================

function sendeJson(response, statusCode, daten) {
  if (response.destroyed || response.writableEnded) {
    return;
  }

  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff"
  });

  response.end(JSON.stringify(daten));
}

// =====================================================
// ANFRAGEKÖRPER LESEN
// =====================================================

async function leseJsonRequest(request) {
  let body = "";

  for await (const chunk of request) {
    body += chunk.toString("utf8");

    if (Buffer.byteLength(body, "utf8") > MAX_BODY_SIZE) {
      throw new Error("Die Anfrage ist zu groß.");
    }
  }

  try {
    return JSON.parse(body);
  } catch {
    throw new Error("Ungültiges JSON in der Anfrage.");
  }
}

// =====================================================
// CHAT-PROTOTYP
//
// WICHTIG:
// Das vorhandene neuronale Netz verarbeitet momentan
// nur zwei Zahlen und lernt die XOR-Aufgabe.
//
// Diese Funktion verwendet vorübergehend einfache Regeln.
// Sie ist KEIN trainiertes Sprachmodell und erzeugt noch
// keine frei formulierten Antworten aus neuronalen Netzen.
// =====================================================

function erzeugeChatAntwort(nachricht) {
  const text = nachricht
    .toLocaleLowerCase("de-DE")
    .trim();

  if (/\b(hallo|hi|hey|servus)\b/.test(text)) {
    return (
      "Hallo! Ich bin dein eigenes KI-Projekt. " +
      "Mein neuronales Netz ist bereits eingebaut, " +
      "lernt momentan aber noch eine mathematische Aufgabe. " +
      "Als Nächstes bringen wir ihm bei, Text zu verarbeiten."
    );
  }

  if (/\b(wer bist du|was bist du)\b/.test(text)) {
    return (
      "Ich bin der Chat-Prototyp deiner selbst programmierten KI. " +
      "Der Server und das neuronale Netz wurden selbst entwickelt. " +
      "Meine Textantworten beruhen momentan noch auf einfachen Regeln, " +
      "nicht auf einem trainierten Sprachmodell."
    );
  }

  if (/\b(hilfe|help|was kannst du|was kannst du machen)\b/.test(text)) {
    return (
      "Ich kann momentan einfache Begrüßungen und einige Fragen " +
      "über dieses Projekt erkennen. Das neuronale Netz verarbeitet " +
      "zwei binäre Eingaben. Für echte Gespräche müssen wir noch " +
      "eine Textkodierung und ein Sprachmodell selbst entwickeln."
    );
  }

  if (/\b(neuron|neuronal|netzwerk|training|gewichte)\b/.test(text)) {
    return (
      "Ein neuronales Netz besteht aus künstlichen Neuronen, " +
      "Verbindungen und Gewichten. Beim Training werden die Gewichte " +
      "anhand von Beispielen angepasst. Unser aktuelles Netz lernt " +
      "XOR. Es versteht noch keine natürliche Sprache."
    );
  }

  if (/\b(tschüss|tschuss|auf wiedersehen|bye)\b/.test(text)) {
    return "Bis bald! Wir können unser eigenes neuronales Netz weiterentwickeln.";
  }

  return (
    "Nachricht empfangen: „" + nachricht.slice(0, 300) + "“\n\n" +
    "Ich kann diesen Text noch nicht wirklich verstehen. " +
    "Mein aktuelles neuronales Netz kann nur zwei Zahlen verarbeiten. " +
    "Als nächsten Entwicklungsschritt bauen wir eine eigene Textkodierung " +
    "und trainieren ein Modell, das Zeichen oder Wörter vorhersagen kann."
  );
}

// =====================================================
// ROUTEN
// =====================================================

async function bearbeiteAnfrage(request, response) {
  let url;

  try {
    url = new URL(
      request.url,
      "http://" + request.headers.host
    );
  } catch {
    sendeJson(response, 400, {
      fehler: "Ungültige URL."
    });
    return;
  }

  const methode = request.method;
  const pfad = url.pathname;

  // ---------------------------------------------------
  // GET /
  // Lädt die getrennte index.html
  // ---------------------------------------------------

  if (methode === "GET" && pfad === "/") {
    if (!fs.existsSync(HTML_FILE)) {
      sendeJson(response, 500, {
        fehler: "index.html wurde im Projektordner nicht gefunden."
      });
      return;
    }

    response.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    });

    response.end(fs.readFileSync(HTML_FILE));
    return;
  }

  // ---------------------------------------------------
  // GET /api/status
  // Gibt den Status des neuronalen Netzes zurück
  // ---------------------------------------------------

  if (methode === "GET" && pfad === "/api/status") {
    sendeJson(response, 200, {
      name: "MeineEigeneKI",
      server: "gestartet",
      trainingsbeispiele: trainingsdaten.length,
      trainingsdurchlaeufe: TRAINING_EPOCHS,
      netz: netz.status(),
      textmodell: false,
      hinweis:
        "Das neuronale Netz kann momentan nur die XOR-Aufgabe lernen."
    });
    return;
  }

  // ---------------------------------------------------
  // POST /api/vorhersage
  // Das neuronale Netz berechnet eine Vorhersage
  // ---------------------------------------------------

  if (methode === "POST" && pfad === "/api/vorhersage") {
    let daten;

    try {
      daten = await leseJsonRequest(request);
    } catch (error) {
      sendeJson(response, 400, {
        fehler: error.message
      });
      return;
    }

    const { x1, x2 } = daten;

    if (
      ![0, 1].includes(x1) ||
      ![0, 1].includes(x2)
    ) {
      sendeJson(response, 400, {
        fehler: "x1 und x2 müssen jeweils 0 oder 1 sein."
      });
      return;
    }

    sendeJson(response, 200, netz.vorhersage(x1, x2));
    return;
  }

  // ---------------------------------------------------
  // POST /api/chat
  // Wird von der Chat-Oberfläche aufgerufen
  // ---------------------------------------------------

  if (methode === "POST" && pfad === "/api/chat") {
    let daten;

    try {
      daten = await leseJsonRequest(request);
    } catch (error) {
      sendeJson(response, 400, {
        fehler: error.message
      });
      return;
    }

    if (
      !daten ||
      typeof daten.message !== "string" ||
      daten.message.trim().length === 0
    ) {
      sendeJson(response, 400, {
        fehler: "Bitte gib eine Nachricht ein."
      });
      return;
    }

    if (daten.message.length > 4000) {
      sendeJson(response, 400, {
        fehler: "Die Nachricht darf höchstens 4000 Zeichen lang sein."
      });
      return;
    }

    const antwort = erzeugeChatAntwort(daten.message.trim());

    sendeJson(response, 200, {
      antwort,
      modus: "regelbasierter-prototyp",
      neuronalesNetzVerwendet:
        false
    });

    return;
  }

  // ---------------------------------------------------
  // Nicht gefundene Route
  // ---------------------------------------------------

  sendeJson(response, 404, {
    fehler: "Diese Route existiert nicht."
  });
}

// =====================================================
// SERVER STARTEN
// =====================================================

const server = http.createServer((request, response) => {
  bearbeiteAnfrage(request, response).catch(error => {
    console.error("Serverfehler:", error);

    sendeJson(response, 500, {
      fehler: "Ein interner Serverfehler ist aufgetreten."
    });
  });
});

server.on("error", error => {
  console.error("Der Server konnte nicht gestartet werden:", error.message);

  if (error.code === "EADDRINUSE") {
    console.error("Port " + PORT + " wird bereits verwendet.");
  }

  process.exitCode = 1;
});

server.listen(PORT, HOST, () => {
  console.log("");
  console.log("======================================");
  console.log("       MEINE EIGENE KI");
  console.log("======================================");
  console.log("Server: http://" + HOST + ":" + PORT);
  console.log("HTML:   " + HTML_FILE);
  console.log("Netz:   netz.js");
  console.log("Daten:  " + DATA_FILE);
  console.log("======================================");
  console.log("Beenden: Strg+C");
  console.log("");
});
