
"use strict";

// ======================================================
// MeineEigeneKI - server.js
// Eigener Server für GitHub und Render
// Keine externen Bibliotheken
// ======================================================

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { NeuronalesNetz } = require("./netz.js");

// Render stellt den Port über process.env.PORT bereit.
const PORT = Number(process.env.PORT) || 3000;
const HOST = "0.0.0.0";

const HTML_DATEI = path.join(__dirname, "index.html");
const DATEN_ORDNER = path.join(__dirname, "Daten");
const TRAININGS_DATEI = path.join(
  DATEN_ORDNER,
  "training.json"
);

const TRAININGS_DURCHLAEUFE = 20000;
const MAX_ANFRAGE = 10000;

// ======================================================
// TRAININGSDATEN LADEN
// ======================================================

function ladeTrainingsdaten() {
  if (!fs.existsSync(TRAININGS_DATEI)) {
    throw new Error(
      "Datei nicht gefunden: Daten/training.json"
    );
  }

  let daten;

  try {
    daten = JSON.parse(
      fs.readFileSync(TRAININGS_DATEI, "utf8")
    );
  } catch (error) {
    throw new Error(
      "training.json kann nicht gelesen werden: " +
      error.message
    );
  }

  if (!Array.isArray(daten) || daten.length === 0) {
    throw new Error(
      "training.json muss Trainingsbeispiele enthalten."
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
        "Ungültiger Trainingseintrag an Position " + i +
        ". Erwartet werden x1, x2 und target mit den Zahlen 0 oder 1."
      );
    }
  }

  console.log(
    "Trainingsdaten geladen:",
    daten.length,
    "Beispiele"
  );

  return daten;
}

// ======================================================
// NEURONALES NETZ ERSTELLEN UND TRAINIEREN
// ======================================================

const trainingsdaten = ladeTrainingsdaten();
const netz = new NeuronalesNetz();

console.log("Das neuronale Netz wird trainiert ...");

const trainingsfehler = netz.trainiere(
  trainingsdaten.slice(),
  TRAININGS_DURCHLAEUFE
);

console.log("Training abgeschlossen.");
console.log("Trainingsdurchläufe:", TRAININGS_DURCHLAEUFE);
console.log("Trainingsfehler:", trainingsfehler);

// ======================================================
// JSON-ANTWORT SENDEN
// ======================================================

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

// ======================================================
// JSON-ANFRAGE LESEN
// ======================================================

function leseJsonAnfrage(request) {
  return new Promise((resolve, reject) => {
    let body = "";
    let zuGross = false;

    request.on("data", teil => {
      if (zuGross) return;

      body += teil.toString("utf8");

      if (Buffer.byteLength(body, "utf8") > MAX_ANFRAGE) {
        zuGross = true;
        body = "";
      }
    });

    request.on("end", () => {
      if (zuGross) {
        reject(new Error("Die Anfrage ist zu groß."));
        return;
      }

      try {
        resolve(JSON.parse(body));
      } catch {
        reject(new Error("Die Anfrage enthält ungültiges JSON."));
      }
    });

    request.on("error", reject);
  });
}

// ======================================================
// EINFACHER CHAT-PROTOTYP
//
// Das aktuelle neuronale Netz lernt XOR mit zwei Zahlen.
// Es kann noch keine natürliche Sprache erzeugen.
//
// Diese Regeln lassen die Chat-Oberfläche funktionieren,
// bis wir ein eigenes Sprachmodell programmieren.
// ======================================================

function chatAntwort(text) {
  const nachricht = text.trim().toLocaleLowerCase("de-DE");

  if (/\b(hallo|hi|hey|servus|guten morgen)\b/.test(nachricht)) {
    return "Hallo! Willkommen bei deiner selbst entwickelten KI. Mein neuronales Netz lernt momentan noch eine mathematische Aufgabe. Wir können es Schritt für Schritt zu einem eigenen Sprachmodell erweitern.";
  }

  if (/\b(wer bist du|was bist du)\b/.test(nachricht)) {
    return "Ich bin der Chat-Prototyp deines KI-Projekts. Mein Server und mein neuronales Netz sind selbst programmiert. Freie Sprachantworten aus dem neuronalen Netz sind noch nicht implementiert.";
  }

  if (/\b(neuronales netz|neuronen|gewichte|backpropagation)\b/.test(nachricht)) {
    return "Ein neuronales Netz berechnet Ausgaben mithilfe von künstlichen Neuronen und Gewichten. Beim Training werden die Gewichte angepasst. Unser aktuelles Netz verwendet eine versteckte Schicht und lernt die XOR-Aufgabe.";
  }

  if (/\b(training|trainieren|lernen)\b/.test(nachricht)) {
    return "Beim Training verarbeitet das Netz Beispiele, berechnet Fehler und passt seine Gewichte an. Deine Beispiele werden aus dem Ordner Daten geladen. Bisher trainieren wir mit Zahlen statt mit Sprache.";
  }

  if (/\b(hilfe|was kannst du|funktionen)\b/.test(nachricht)) {
    return "Ich kann momentan einfache Begrüßungen und Fragen über mein Projekt erkennen. Die mathematischen Vorhersagen meines neuronalen Netzes sind über /api/vorhersage verfügbar. Für echte Gespräche müssen wir ein eigenes Sprachmodell entwickeln.";
  }

  if (/\b(tschüss|tschuss|auf wiedersehen|bye)\b/.test(nachricht)) {
    return "Bis bald! Wir entwickeln unser eigenes neuronales Netz Schritt für Schritt weiter.";
  }

  return (
    "Ich habe deine Nachricht erhalten: „" +
    text.slice(0, 300) +
    "“\n\nIch kann diesen Text noch nicht wirklich verstehen. " +
    "Mein neuronales Netz verarbeitet derzeit zwei Zahlen und lernt eine mathematische Aufgabe. " +
    "Als Nächstes müssen wir eine eigene Textkodierung und ein Sprachmodell entwickeln."
  );
}

// ======================================================
// ANFRAGEN VERARBEITEN
// ======================================================

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

  // ----------------------------------------------------
  // CHAT-WEBSEITE
  // GET /
  // ----------------------------------------------------

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

  // ----------------------------------------------------
  // STATUS DES NEURONALEN NETZES
  // GET /api/status
  // ----------------------------------------------------

  if (methode === "GET" && route === "/api/status") {
    sendeJson(response, 200, {
      name: "MeineEigeneKI",
      server: "online",
      trainingsbeispiele: trainingsdaten.length,
      netz: netz.status(),
      trainingsfehler,
      sprachmodellVorhanden: false,
      hinweis: "Das neuronale Netz lernt momentan XOR."
    });
    return;
  }

  // ----------------------------------------------------
  // NEURONALE VORHERSAGE
  // POST /api/vorhersage
  // ----------------------------------------------------

  if (methode === "POST" && route === "/api/vorhersage") {
    let daten;

    try {
      daten = await leseJsonAnfrage(request);
    } catch (error) {
      sendeJson(response, 400, {
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

  // ----------------------------------------------------
  // CHAT
  // POST /api/chat
  // ----------------------------------------------------

  if (methode === "POST" && route === "/api/chat") {
    let daten;

    try {
      daten = await leseJsonAnfrage(request);
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
        fehler: "Die Nachricht darf maximal 4000 Zeichen enthalten."
      });
      return;
    }

    sendeJson(response, 200, {
      antwort: chatAntwort(daten.message),
      modus: "regelbasierter-prototyp",
      neuronalesNetzVerwendet: false
    });

    return;
  }

  // ----------------------------------------------------
  // NICHT GEFUNDENE ROUTE
  // ----------------------------------------------------

  sendeJson(response, 404, {
    fehler: "Route nicht gefunden."
  });
}

// ======================================================
// SERVER STARTEN
// ======================================================

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

server.on("error", error => {
  console.error("Serverstart fehlgeschlagen:", error.message);

  if (error.code === "EADDRINUSE") {
    console.error("Der Port " + PORT + " ist bereits belegt.");
  }

  process.exitCode = 1;
});

server.listen(PORT, HOST, () => {
  console.log("");
  console.log("==================================");
  console.log("       MEINE EIGENE KI");
  console.log("==================================");
  console.log("Server-Port:", PORT);
  console.log("HTML:", HTML_DATEI);
  console.log("Netzwerk: netz.js");
  console.log("Trainingsdaten:", TRAININGS_DATEI);
  console.log("Trainingsbeispiele:", trainingsdaten.length);
  console.log("==================================");
});
