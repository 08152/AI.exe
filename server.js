
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
// GitHub-Konfiguration aus Render
// --------------------------------------------------

const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const GITHUB_OWNER = process.env.GITHUB_OWNER;
const GITHUB_REPO = process.env.GITHUB_REPO;
const GITHUB_BRANCH = process.env.GITHUB_BRANCH || "main";

function githubIstKonfiguriert() {
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
  const ergebnis = [];
  const dateinamen = [];

  const dateien = fs.readdirSync(DATEN_ORDNER, {
    withFileTypes: true
  })
    .filter(datei =>
      datei.isFile() &&
      datei.name.toLowerCase().endsWith(".json") &&
      datei.name.toLowerCase() !== "tokenizer.json"
    )
    .sort((a, b) => a.name.localeCompare(b.name));

  for (const datei of dateien) {
    const dateipfad = path.join(
      DATEN_ORDNER,
      datei.name
    );

    try {
      const daten = JSON.parse(
        fs.readFileSync(dateipfad, "utf8")
      );

      ergebnis.push(daten);
      dateinamen.push(datei.name);

      console.log("Trainingsdatei geladen:", datei.name);
    } catch (fehler) {
      console.error(
        "JSON-Datei übersprungen:",
        datei.name,
        fehler.message
      );
    }
  }

  console.log("Geladene Trainingsdateien:", dateinamen.length);

  return {
    daten: ergebnis,
    dateinamen
  };
}

// Der aktuelle Tokenizer ist kleinschreibungssensitiv.
// Darum normalisieren wir Trainings-Texte und Chat-Eingaben.
function normalisiereTexte(daten) {
  if (typeof daten === "string") {
    return daten
      .normalize("NFC")
      .toLocaleLowerCase("de-DE");
  }

  if (Array.isArray(daten)) {
    return daten.map(normalisiereTexte);
  }

  if (daten && typeof daten === "object") {
    const neu = {};

    for (const [key, wert] of Object.entries(daten)) {
      neu[key] = normalisiereTexte(wert);
    }

    return neu;
  }

  return daten;
}

// --------------------------------------------------
// Tokenizer vorbereiten
// --------------------------------------------------

// Das Vokabular wird bei jedem Start aus den aktuellen
// Trainingsdaten neu aufgebaut. So vermeiden wir alte,
// nicht passende Token-IDs aus vorherigen Versionen.
const tokenizer = new Tokenizer();

// --------------------------------------------------
// Das neuronale Sprachmodell aufbauen und trainieren
// --------------------------------------------------

const { daten: roheTrainingsdaten, dateinamen } =
  ladeTrainingsdaten();

const trainingsdaten = normalisiereTexte(
  roheTrainingsdaten
);

const netz = new NeuronalesNetz(tokenizer, {
  maxVokabular: 256,
  embeddingGroesse: 8,
  versteckteNeuronen: 16,
  kontextLaenge: 12
});

let trainingsFehler = null;
let trainingsStatus = null;

if (trainingsdaten.length === 0) {
  trainingsFehler =
    "Keine JSON-Trainingsdateien in Daten/ gefunden.";
} else {
  try {
    trainingsStatus = netz.trainiereTexte(
      trainingsdaten,
      tokenizer,
      {
        epochen: Number(process.env.TRAINING_EPOCHS) || 4,
        maxTrainingsBeispiele: 1000,
        lernrate: 0.025
      }
    );

    console.log("Training abgeschlossen.");
    console.log("Modellstatus:", trainingsStatus);
  } catch (fehler) {
    trainingsFehler = fehler.message;

    console.error(
      "Das Sprachmodell konnte nicht trainiert werden:",
      fehler.message
    );
  }
}

// Tokenizer lokal speichern.
try {
  tokenizer.speichern(TOKENIZER_DATEI);
  console.log("Tokenizer lokal gespeichert:", TOKENIZER_DATEI);
} catch (fehler) {
  console.error(
    "Tokenizer konnte nicht gespeichert werden:",
    fehler.message
  );
}

// --------------------------------------------------
// GitHub: Tokenizer-Datei erstellen oder aktualisieren
// --------------------------------------------------

async function synchronisiereTokenizerMitGitHub() {
  if (!githubIstKonfiguriert()) {
    console.log(
      "GitHub-Upload übersprungen: " +
      "GITHUB_TOKEN, GITHUB_OWNER oder GITHUB_REPO fehlt."
    );

    return;
  }

  if (!netz.bereit) {
    console.log(
      "GitHub-Upload übersprungen: " +
      "Das Sprachmodell wurde nicht trainiert."
    );

    return;
  }

  if (typeof fetch !== "function") {
    console.error(
      "GitHub-Upload benötigt Node.js 18 oder neuer."
    );

    return;
  }

  const relativerPfad = "modelle/tokenizer.json";

  const githubUrl =
    "https://api.github.com/repos/" +
    encodeURIComponent(GITHUB_OWNER) + "/" +
    encodeURIComponent(GITHUB_REPO) + "/contents/" +
    relativerPfad.split("/")
      .map(encodeURIComponent)
      .join("/");

  const header = {
    "Accept": "application/vnd.github+json",
    "Authorization": `Bearer ${GITHUB_TOKEN}`,
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "MeineEigeneKI"
  };

  const dateiInhalt = fs.readFileSync(
    TOKENIZER_DATEI,
    "utf8"
  );

  try {
    // Prüfen, ob die Datei schon existiert.
    const getAntwort = await fetch(
      githubUrl + "?ref=" +
      encodeURIComponent(GITHUB_BRANCH),
      { headers: header }
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
          console.log("GitHub-Tokenizer ist bereits aktuell.");
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

    // GitHub verlangt den bisherigen SHA beim Aktualisieren.
    if (sha) {
      payload.sha = sha;
    }

    const putAntwort = await fetch(githubUrl, {
      method: "PUT",
      headers: {
        ...header,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const putDaten = await putAntwort.json().catch(() => ({}));

    if (!putAntwort.ok) {
      throw new Error(
        `GitHub-Upload fehlgeschlagen: HTTP ${putAntwort.status}. ` +
        `${putDaten.message || "Berechtigungen und Branch prüfen."}`
      );
    }

    console.log(
      "Tokenizer wurde im GitHub-Repository gespeichert."
    );

    if (putDaten.content && putDaten.content.html_url) {
      console.log("Datei:", putDaten.content.html_url);
    }
  } catch (fehler) {
    // Niemals den GitHub-Token ausgeben.
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
    let abgeschlossen = false;

    req.setEncoding("utf8");

    req.on("data", teil => {
      if (abgeschlossen) return;

      bytes += Buffer.byteLength(teil, "utf8");

      if (bytes > maximalBytes) {
        abgeschlossen = true;

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
      if (abgeschlossen) return;

      abgeschlossen = true;

      try {
        resolve(JSON.parse(inhalt || "{}"));
      } catch {
        const fehler = new Error("Ungültiges JSON.");
        fehler.status = 400;
        reject(fehler);
      }
    });

    req.on("error", fehler => {
      if (abgeschlossen) return;

      abgeschlossen = true;
      reject(fehler);
    });
  });
}

// --------------------------------------------------
// HTTP-Server
// --------------------------------------------------

const server = http.createServer(async (req, res) => {
  const host = req.headers.host || "localhost";

  let url;

  try {
    url = new URL(req.url, `http://${host}`);
  } catch {
    return sendeJson(res, 400, {
      fehler: "Ungültige URL."
    });
  }

  try {
    // Website ausliefern.
    if (
      req.method === "GET" &&
      (url.pathname === "/" || url.pathname === "/index.html")
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

    // Status des Sprach-Hirns anzeigen.
    if (
      req.method === "GET" &&
      url.pathname === "/api/status"
    ) {
      return sendeJson(res, 200, {
        ok: true,
        modell: netz.status(),
        tokenizer: tokenizer.status(),
        trainingsFehler,
        trainingsdateien: dateinamen,
        githubUploadKonfiguriert: githubIstKonfiguriert()
      });
    }

    // Antwort mit dem neuronalen Sprachmodell erzeugen.
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

      if (!netz.bereit) {
        return sendeJson(res, 503, {
          fehler:
            trainingsFehler ||
            "Das neuronale Sprachmodell ist noch nicht bereit."
        });
      }

      // Eingabe in dieselbe Schreibweise bringen wie die Trainingsdaten.
      const eingabe = nachricht
        .normalize("NFC")
        .toLocaleLowerCase("de-DE");

      const antwort = netz.antwortGenerieren(eingabe, {
        maxTokens: 45,
        temperatur: 0.8,
        topK: 8
      });

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
    console.error("Anfrage fehlgeschlagen:", fehler.message);

    return sendeJson(
      res,
      fehler.status || 500,
      { fehler: fehler.message }
    );
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Server läuft auf Port ${PORT}.`);
  console.log("Sprachmodell bereit:", netz.bereit);

  // Datei nach dem Start mit GitHub synchronisieren.
  void synchronisiereTokenizerMitGitHub();
});
