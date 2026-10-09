
"use strict";

// ============================================================
// MeineEigeneKI - server.js
// GitHub + Render
// Eigener HTTP-Server ohne externe Bibliotheken
//
// Projektstruktur:
//   server.js
//   netz.js
//   index.html
//   package.json
//   Daten/
//     xor.json
//     weitere.json
// ============================================================

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const { NeuronalesNetz } = require("./netz.js");

// ============================================================
// KONFIGURATION
// ============================================================

const PORT = Number(process.env.PORT) || 3000;
const HOST = "0.0.0.0";

const HTML_DATEI = path.join(__dirname, "index.html");
const DATEN_ORDNER = path.join(__dirname, "Daten");

const TRAININGS_DURCHLAEUFE =
  Number(process.env.TRAINING_EPOCHS) || 20000;

const MAX_ANFRAGE_GROESSE = 10000;

// ============================================================
// ALLE JSON-DATEIEN AUS DEM DATEN-ORDNER LADEN
// ============================================================

function ladeTrainingsdaten() {
  if (!fs.existsSync(DATEN_ORDNER)) {
    throw new Error(
      'Der Ordner "Daten" wurde nicht gefunden: ' + DATEN_ORDNER
    );
  }

  const dateien = fs.readdirSync(DATEN_ORDNER, {
    withFileTypes: true
  })
    .filter(datei =>
      datei.isFile() &&
      datei.name.toLowerCase().endsWith(".json")
    )
    .map(datei => datei.name)
    .sort((a, b) => a.localeCompare(b));

  if (dateien.length === 0) {
    throw new Error(
      'Im Ordner "Daten" wurden keine JSON-Dateien gefunden.'
    );
  }

  const alleBeispiele = [];

  console.log("");
  console.log("======================================");
  console.log("       TRAININGSDATEN LADEN");
  console.log("======================================");

  for (const dateiname of dateien) {
    const dateipfad = path.join(DATEN_ORDNER, dateiname);

    let inhalt;

    try {
      inhalt = JSON.parse(
        fs.readFileSync(dateipfad, "utf8")
      );
    } catch (error) {
      throw new Error(
        'Fehler in "' + dateiname + '": ' + error.message
      );
    }

    // Eine JSON-Datei kann entweder ein Array mit mehreren
    // Beispielen oder ein einzelnes Beispiel enthalten.
    const beispiele = Array.isArray(inhalt)
      ? inhalt
      : [inhalt];

    if (beispiele.length === 0) {
      console.warn(
        "Leere JSON-Datei übersprungen:",
        dateiname
      );
      continue;
    }

    for (let i = 0; i < beispiele.length; i++) {
      const beispiel = beispiele[i];

      // Die aktuelle netz.js unterstützt zwei binäre Eingaben
      // und ein binäres Trainingsziel.
      if (
        !beispiel ||
        typeof beispiel !== "object" ||
        ![0, 1].includes(beispiel.x1) ||
        ![0, 1].includes(beispiel.x2) ||
        ![0, 1].includes(beispiel.target)
      ) {
        throw new Error(
          'Ungültiges Trainingsbeispiel in "' +
          dateiname + '", Eintrag ' + (i + 1) +
          '. Erwartet werden x1, x2 und target, ' +
          'jeweils mit dem Zahlenwert 0 oder 1.'
        );
      }

      alleBeispiele.push({
        x1: beispiel.x1,
        x2: beispiel.x2,
        target: beispiel.target
      });
    }

    console.log(
      "Geladen: " + dateiname +
      " | Beispiele: " + beispiele.length
    );
  }

  if (alleBeispiele.length === 0) {
    throw new Error(
      "In den JSON-Dateien wurden keine Trainingsbeispiele gefunden."
    );
  }

  console.log("--------------------------------------");
  console.log("JSON-Dateien gefunden:", dateien.length);
  console.log("Trainingsbeispiele insgesamt:", alleBeispiele.length);
  console.log("======================================");
  console.log("");

  return {
    dateien,
    beispiele: alleBeispiele
  };
}

// ============================================================
// NEURONALES NETZ INITIALISIEREN UND TRAINIEREN
// ============================================================

const trainingspaket = ladeTrainingsdaten();
const trainingsdaten = trainingspaket.beispiele;

const netz = new NeuronalesNetz();

console.log("Trainiere das eigene neuronale Netz ...");

const trainingsfehler = netz.trainiere(
  trainingsdaten.slice(),
  TRAININGS_DURCHLAEUFE
);

console.log("Training abgeschlossen.");
console.log("Trainingsdurchläufe:", TRAININGS_DURCHLAEUFE);
console.log("Trainingsfehler:", trainingsfehler);
console.log("");

// ============================================================
// JSON-ANTWORT SENDEN
// ============================================================

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

// ============================================================
// JSON-ANFRAGE EINLESEN
// ============================================================

function leseJsonAnfrage(request) {
  return new Promise((resolve, reject) => {
    let body = "";
    let groesse = 0;
    let zuGross = false;

    request.on("data", teil => {
      groesse += teil.length;

      if (groesse > MAX_ANFRAGE_GROESSE) {
        zuGross = true;
        body = "";
        return;
      }

      if (!zuGross) {
        body += teil.toString("utf8");
      }
    });

    request.on("end", () => {
      if (zuGross) {
        const error = new Error(
          "Die Anfrage ist zu groß. Maximal 10000 Bytes erlaubt."
        );

        error.statusCode = 413;
        reject(error);
        return;
      }

      try {
        resolve(JSON.parse(body));
      } catch {
        const error = new Error(
          "Die Anfrage enthält ungültiges JSON."
        );

        error.statusCode = 400;
        reject(error);
      }
    });

    request.on("error", reject);
  });
}

// ============================================================
// CHAT-ANTWORTEN
//
// WICHTIG:
// Das bestehende neuronale Netz kann nur zwei Zahlen
// verarbeiten. Die Chat-Antworten sind daher zunächst
// regelbasiert. Sie werden NICHT vom neuronalen Netz erzeugt.
// ============================================================

function erzeugeChatAntwort(nachricht) {
  const text = nachricht
    .trim()
    .toLocaleLowerCase("de-DE");

  if (/\b(hallo|hi|hey|servus)\b/.test(text)) {
    return (
      "Hallo! Willkommen bei deiner selbst entwickelten KI. " +
      "Mein neuronales Netz wird mit den Trainingsdaten aus " +
      'dem Ordner "Daten" trainiert. Es kann derzeit aber ' +
      "noch keine natürliche Sprache erzeugen."
    );
  }

  if (/\b(wer bist du|was bist du)\b/.test(text)) {
    return (
      "Ich bin der Chat-Prototyp deines eigenen KI-Projekts. " +
      "Mein Server und mein neuronales Netz wurden selbst " +
      "programmiert. Meine aktuellen Chat-Antworten basieren " +
      "noch auf einfachen Regeln."
    );
  }

  if (/\b(neuron|neuronal|gewichte|backpropagation)\b/.test(text)) {
    return (
      "Ein neuronales Netz besteht aus künstlichen Neuronen, " +
      "Gewichten und Berechnungen. Beim Training werden die " +
      "Gewichte anhand von Beispielen angepasst. Unser " +
      "aktuelles Netz verarbeitet zwei binäre Eingaben."
    );
  }

  if (/\b(training|trainieren|trainingsdaten|lernen)\b/.test(text)) {
    return (
      "Beim Serverstart werden alle JSON-Dateien direkt aus " +
      'dem Ordner "Daten" eingelesen. Die gültigen Beispiele ' +
      "werden zusammengeführt und an das neuronale Netz " +
      "übergeben. Derzeit muss jedes Beispiel x1, x2 und " +
      "target mit den Werten 0 oder 1 enthalten."
    );
  }

  if (/\b(hilfe|was kannst du|funktionen)\b/.test(text)) {
    return (
      "Ich kann momentan einige einfache Fragen über das " +
      "KI-Projekt beantworten. Über /api/status kannst du " +
      "den Trainingsstatus abrufen. Über /api/vorhersage " +
      "kannst du das neuronale Netz mit zwei Zahlen testen."
    );
  }

  if (/\b(tschüss|tschuss|auf wiedersehen|bye)\b/.test(text)) {
    return (
      "Bis bald! Wir können dein eigenes neuronales Netz " +
      "Schritt für Schritt erweitern."
    );
  }

  return (
    "Ich habe deine Nachricht erhalten: „" +
    nachricht.slice(0, 300) +
    "“\n\n" +
    "Ich kann den Text momentan noch nicht wirklich verstehen. " +
    "Mein neuronales Netz verarbeitet bislang zwei Zahlen. " +
    "Für echte Sprachfähigkeiten müssen wir selbst eine " +
    "Textkodierung und ein Sprachmodell entwickeln."
  );
}

// ============================================================
// HTTP-ROUTEN
// ============================================================

async function verarbeiteAnfrage(request, response) {
  let url;

  try {
    url = new URL(
      request.url,
      "http://localhost:" + PORT
    );
  } catch {
    sendeJson(response, 400, {
      fehler: "Ungültige URL."
    });
    return;
  }

  const methode = request.method;
  const route = url.pathname;

  // ----------------------------------------------------------
  // GET /
  // Liefert die Chat-Oberfläche index.html aus.
  // ----------------------------------------------------------

  if (methode === "GET" && route === "/") {
    let html;

    try {
      html = fs.readFileSync(HTML_DATEI);
    } catch {
      sendeJson(response, 500, {
        fehler: "index.html wurde nicht gefunden."
      });
      return;
    }

    response.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    });

    response.end(html);
    return;
  }

  // ----------------------------------------------------------
  // GET /api/status
  // Gibt Informationen über die geladenen Daten und das Netz.
  // ----------------------------------------------------------

  if (methode === "GET" && route === "/api/status") {
    sendeJson(response, 200, {
      name: "MeineEigeneKI",
      server: "online",

      trainingsdateien: trainingspaket.dateien,
      anzahlTrainingsdateien: trainingspaket.dateien.length,
      trainingsbeispiele: trainingsdaten.length,

      trainingsdurchlaeufe: TRAININGS_DURCHLAEUFE,
      trainingsfehler,

      netz: netz.status(),

      sprachmodellVorhanden: false,

      hinweis:
        "Das Netz lernt momentan eine mathematische Aufgabe. " +
        "Die Chat-Antworten sind noch regelbasiert."
    });

    return;
  }

  // ----------------------------------------------------------
  // POST /api/vorhersage
  // Lässt das neuronale Netz eine Vorhersage berechnen.
  //
  // Beispiel:
  // {"x1":0,"x2":1}
  // ----------------------------------------------------------

  if (methode === "POST" && route === "/api/vorhersage") {
    let daten;

    try {
      daten = await leseJsonAnfrage(request);
    } catch (error) {
      sendeJson(response, error.statusCode || 400, {
        fehler: error.message
      });
      return;
    }

    const { x1, x2 } = daten || {};

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

  // ----------------------------------------------------------
  // POST /api/chat
  // Kompatibel mit der bisherigen index.html.
  // ----------------------------------------------------------

  if (methode === "POST" && route === "/api/chat") {
    let daten;

    try {
      daten = await leseJsonAnfrage(request);
    } catch (error) {
      sendeJson(response, error.statusCode || 400, {
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
        fehler: "Die Nachricht darf höchstens 4000 Zeichen enthalten."
      });
      return;
    }

    const antwort = erzeugeChatAntwort(
      daten.message.trim()
    );

    sendeJson(response, 200, {
      antwort,
      modus: "regelbasierter-prototyp",
      neuronalesNetzVerwendet: false
    });

    return;
  }

  // ----------------------------------------------------------
  // ALLE ANDEREN ROUTEN
  // ----------------------------------------------------------

  sendeJson(response, 404, {
    fehler: "Route nicht gefunden."
  });
}

// ============================================================
// SERVER ERSTELLEN
// ============================================================

const server = http.createServer((request, response) => {
  verarbeiteAnfrage(request, response).catch(error => {
    console.error("Anfragefehler:", error);

    if (!response.headersSent) {
      sendeJson(response, 500, {
        fehler: "Interner Serverfehler."
      });
    } else if (!response.destroyed) {
      response.destroy();
    }
  });
});

// ============================================================
// FEHLER BEIM SERVERSTART
// ============================================================

server.on("error", error => {
  console.error("Serverfehler:", error.message);

  if (error.code === "EADDRINUSE") {
    console.error("Port " + PORT + " ist bereits belegt.");
  }

  process.exitCode = 1;
});

// ============================================================
// SERVER STARTEN
// ============================================================

server.listen(PORT, HOST, () => {
  console.log("");
  console.log("============================================");
  console.log("          MEINE EIGENE KI");
  console.log("============================================");
  console.log("Server:", "http://" + HOST + ":" + PORT);
  console.log("HTML-Datei:", HTML_DATEI);
  console.log("Neuronales Netz: netz.js");
  console.log("Datenordner:", DATEN_ORDNER);
  console.log("JSON-Dateien:", trainingspaket.dateien.length);
  console.log("Trainingsbeispiele:", trainingsdaten.length);
  console.log("Trainingsdurchläufe:", TRAININGS_DURCHLAEUFE);
  console.log("============================================");
  console.log("Die Chat-Oberfläche ist bereit.");
  console.log("");
});
