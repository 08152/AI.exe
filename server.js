
"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const { Tokenizer } = require("./tokenizer.js");
const { NeuronalesNetz } = require("./netz.js");

const PORT = Number(process.env.PORT) || 3000;

const ROOT = __dirname;
const DATEN_ORDNER = path.join(ROOT, "Daten");
const MODELL_ORDNER = path.join(ROOT, "modelle");
const HTML_DATEI = path.join(ROOT, "index.html");
const TOKENIZER_DATEI = path.join(
  MODELL_ORDNER,
  "tokenizer.json"
);

fs.mkdirSync(DATEN_ORDNER, { recursive: true });
fs.mkdirSync(MODELL_ORDNER, { recursive: true });

// --------------------------------------------------
// GitHub-Konfiguration
// --------------------------------------------------

const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const GITHUB_OWNER = process.env.GITHUB_OWNER;
const GITHUB_REPO = process.env.GITHUB_REPO;
const GITHUB_BRANCH = process.env.GITHUB_BRANCH || "main";

function githubKonfiguriert() {
  return Boolean(
    GITHUB_TOKEN &&
    GITHUB_OWNER &&
    GITHUB_REPO
  );
}

// --------------------------------------------------
// Trainingsdaten laden
// --------------------------------------------------

function ladeTrainingsdaten() {
  const dateien = fs.readdirSync(DATEN_ORDNER, {
    withFileTypes: true
  }).filter(datei =>
    datei.isFile() &&
    datei.name.toLowerCase().endsWith(".json") &&
    datei.name.toLowerCase() !== "tokenizer.json"
  ).sort((a, b) => a.name.localeCompare(b.name));

  const daten = [];
  const geladeneDateien = [];
  const fehlerDateien = [];

  for (const datei of dateien) {
    const dateipfad = path.join(
      DATEN_ORDNER,
      datei.name
    );

    try {
      const inhalt = fs.readFileSync(
        dateipfad,
        "utf8"
      );

      const json = JSON.parse(inhalt);

      daten.push(json);
      geladeneDateien.push(datei.name);

      console.log("Trainingsdatei geladen:", datei.name);
    } catch (fehler) {
      fehlerDateien.push(datei.name);

      console.error(
        `Datei ${datei.name} übersprungen:`,
        fehler.message
      );
    }
  }

  return {
    daten,
    geladeneDateien,
    fehlerDateien
  };
}

// --------------------------------------------------
// Tokenizer und neuronales Sprachmodell
// --------------------------------------------------

let tokenizer = new Tokenizer();

let netz = new NeuronalesNetz(tokenizer, {
  maxVokabular: 256,
  embeddingGroesse: 8,
  versteckteNeuronen: 16,
  kontextLaenge: 12
});

let trainingsBereit = false;
let trainingsFehler = null;
let trainingsStatus = null;

let geladeneDateien = [];
let fehlerDateien = [];

function trainiereSprachmodell() {
  trainingsBereit = false;
  trainingsFehler = null;
  trainingsStatus = null;

  const ergebnis = ladeTrainingsdaten();

  geladeneDateien = ergebnis.geladeneDateien;
  fehlerDateien = ergebnis.fehlerDateien;

  if (ergebnis.daten.length === 0) {
    trainingsFehler =
      "Keine gültigen JSON-Trainingsdateien in Daten/ gefunden.";

    console.error(trainingsFehler);
    return false;
  }

  try {
    // Jede Trainingsdatei kann Frage-Antwort-Paare,
    // Textsammlungen oder einfache Textfelder enthalten.
    trainingsStatus = netz.trainiereTexte(
      ergebnis.daten,
      tokenizer,
      {
        epochen: Number(process.env.TRAINING_EPOCHS) || 4,
        maxTrainingsBeispiele: 1000,
        lernrate: 0.025
      }
    );

    trainingsBereit = netz.bereit;

    if (!trainingsBereit) {
      throw new Error(
        "Das neuronale Netz meldet sich nicht als bereit."
      );
    }

    console.log("Training abgeschlossen.");
    console.log("Sprachmodell:", trainingsStatus);

    // tokenizer.js speichert das Vokabular bereits
    // während trainiereTexte(). Hier zusätzlich absichern.
    tokenizer.speichern(TOKENIZER_DATEI);

    return true;
  } catch (fehler) {
    trainingsFehler = fehler.message;

    console.error(
      "Training fehlgeschlagen:",
      fehler.message
    );

    return false;
  }
}

// Beim Start wird das Modell aus Daten/ trainiert.
trainiereSprachmodell();

// --------------------------------------------------
// Tokenizer mit GitHub synchronisieren
// --------------------------------------------------

async function synchronisiereTokenizerMitGitHub() {
  if (!githubKonfiguriert()) {
    console.log(
      "GitHub-Synchronisierung übersprungen: " +
      "GITHUB_TOKEN, GITHUB_OWNER oder GITHUB_REPO fehlt."
    );
    return;
  }

  if (!trainingsBereit) {
    console.log(
      "GitHub-Synchronisierung übersprungen: " +
      "das Sprachmodell ist nicht trainiert."
    );
    return;
  }

  if (typeof fetch !== "function") {
    console.error(
      "GitHub-API benötigt Node.js 18 oder neuer."
    );
    return;
  }

  const dateipfad = "modelle/tokenizer.json";

  const url =
    "https://api.github.com/repos/" +
    encodeURIComponent(GITHUB_OWNER) +
    "/" +
    encodeURIComponent(GITHUB_REPO) +
    "/contents/" +
    dateipfad
      .split("/")
      .map(encodeURIComponent)
      .join("/");

  const headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": `Bearer ${GITHUB_TOKEN}`,
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "MeineEigeneKI"
  };

  try {
    const dateiInhalt = fs.readFileSync(
      TOKENIZER_DATEI,
      "utf8"
    );

    // Vorhandene Datei abfragen, um ihren SHA zu erhalten.
    const getAntwort = await fetch(
      url + "?ref=" + encodeURIComponent(GITHUB_BRANCH),
      { headers }
    );

    let sha = null;

    if (getAntwort.status === 200) {
      const vorhandeneDatei = await getAntwort.json();

      sha = vorhandeneDatei.sha;

      if (
        vorhandeneDatei.encoding === "base64" &&
        typeof vorhandeneDatei.content === "string"
      ) {
        const alterInhalt = Buffer.from(
          vorhandeneDatei.content.replace(/\s/g, ""),
          "base64"
        ).toString("utf8");

        if (alterInhalt === dateiInhalt) {
          console.log(
            "GitHub-Tokenizer ist bereits aktuell."
          );
          return;
        }
      }
    } else if (getAntwort.status !== 404) {
      throw new Error(
        `GitHub-Dateiabfrage fehlgeschlagen: HTTP ${getAntwort.status}`
      );
    }

    const payload = {
      message: "Tokenizer aus Trainingsdaten aktualisieren",
      content: Buffer.from(
        dateiInhalt,
        "utf8"
      ).toString("base64"),
      branch: GITHUB_BRANCH
    };

    if (sha) {
      payload.sha = sha;
    }

    const putAntwort = await fetch(url, {
      method: "PUT",
      headers: {
        ...headers,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const putDaten = await putAntwort.json().catch(
      () => ({})
    );

    if (!putAntwort.ok) {
      throw new Error(
        `GitHub-Upload fehlgeschlagen: HTTP ${putAntwort.status}. ` +
        `${putDaten.message || "Repository und Berechtigungen prüfen."}`
      );
    }

    console.log(
      "Tokenizer wurde in GitHub gespeichert."
    );

    if (putDaten.content?.html_url) {
      console.log(
        "Tokenizer-Datei:",
        putDaten.content.html_url
      );
    }
  } catch (fehler) {
    // Das Token niemals ausgeben.
    console.error(
      "GitHub-Synchronisierung fehlgeschlagen:",
      fehler.message
    );
  }
}

// --------------------------------------------------
// HTTP-Hilfsfunktionen
// --------------------------------------------------

function sendeJson(res, status, daten) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store"
  });

  res.end(JSON.stringify(daten));
}

function leseJson(req, maximalBytes = 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let inhalt = "";
    let bytes = 0;
    let beendet = false;
    let zuGross = false;

    req.setEncoding("utf8");

    req.on("data", teil => {
      if (beendet || zuGross) return;

      bytes += Buffer.byteLength(teil, "utf8");

      if (bytes > maximalBytes) {
        zuGross = true;

        const fehler = new Error(
          "Die Anfrage ist zu groß."
        );

        fehler.status = 413;
        reject(fehler);
        return;
      }

      inhalt += teil;
    });

    req.on("end", () => {
      if (beendet || zuGross) return;

      beendet = true;

      try {
        resolve(JSON.parse(inhalt || "{}"));
      } catch {
        const fehler = new Error(
          "Die Anfrage enthält ungültiges JSON."
        );

        fehler.status = 400;
        reject(fehler);
      }
    });

    req.on("error", fehler => {
      if (beendet || zuGross) return;

      beendet = true;
      reject(fehler);
    });
  });
}

// --------------------------------------------------
// HTTP-Server
// --------------------------------------------------

const server = http.createServer(async (req, res) => {
  let url;

  try {
    url = new URL(
      req.url,
      `http://${req.headers.host || "localhost"}`
    );
  } catch {
    return sendeJson(res, 400, {
      fehler: "Ungültige URL."
    });
  }

  try {
    // Website
    if (
      req.method === "GET" &&
      (url.pathname === "/" ||
       url.pathname === "/index.html")
    ) {
      if (!fs.existsSync(HTML_DATEI)) {
        return sendeJson(res, 404, {
          fehler: "index.html wurde nicht gefunden."
        });
      }

      res.writeHead(200, {
        "Content-Type": "text/html; charset=utf-8"
      });

      return res.end(
        fs.readFileSync(HTML_DATEI)
      );
    }

    // Status des Modells
    if (
      req.method === "GET" &&
      url.pathname === "/api/status"
    ) {
      return sendeJson(res, 200, {
        ok: true,
        modell: netz.status(),
        tokenizer: tokenizer.status(),
        trainingsBereit,
        trainingsFehler,
        trainingsdateien: geladeneDateien,
        uebersprungeneDateien: fehlerDateien,
        githubSynchronisierungKonfiguriert:
          githubKonfiguriert()
      });
    }

    // Chat: Die Antwort wird durch das neuronale Netz erzeugt.
    if (
      req.method === "POST" &&
      url.pathname === "/api/chat"
    ) {
      const daten = await leseJson(req);

      const nachricht =
        typeof daten.nachricht === "string"
          ? daten.nachricht
          : typeof daten.message === "string"
            ? daten.message
            : typeof daten.text === "string"
              ? daten.text
              : "";

      if (!nachricht.trim()) {
        return sendeJson(res, 400, {
          fehler: "Bitte gib eine Nachricht ein."
        });
      }

      if (!trainingsBereit || !netz.bereit) {
        return sendeJson(res, 503, {
          fehler:
            trainingsFehler ||
            "Das neuronale Sprachmodell ist noch nicht bereit."
        });
      }

      const antwort = netz.antwortGenerieren(
        nachricht,
        {
          maxTokens: 45,
          temperatur: 0.65,
          topK: 5
        }
      );

      return sendeJson(res, 200, {
        antwort,
        reply: antwort,
        generiertVomNeuronalenNetz: true
      });
    }

    return sendeJson(res, 404, {
      fehler: "API-Endpunkt nicht gefunden."
    });
  } catch (fehler) {
    console.error(
      "Anfrage fehlgeschlagen:",
      fehler.message
    );

    return sendeJson(res, fehler.status || 500, {
      fehler: fehler.message
    });
  }
});

// --------------------------------------------------
// Server starten
// --------------------------------------------------

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Server läuft auf Port ${PORT}.`);
  console.log("Sprachmodell bereit:", trainingsBereit);

  // Die GitHub-Synchronisierung blockiert den Chatstart nicht.
  void synchronisiereTokenizerMitGitHub();
});
