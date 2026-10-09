
"use strict";

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const netzModul = require("./netz.js");
const { Tokenizer } = require("./tokenizer.js");

const NeuronalesNetz =
  netzModul.NeuronalesNetz || netzModul;

const PORT = Number(process.env.PORT) || 3000;
const ROOT = __dirname;
const DATEN_ORDNER = path.join(ROOT, "Daten");
const MODELL_ORDNER = path.join(ROOT, "modelle");
const HTML_DATEI = path.join(ROOT, "index.html");
const TOKENIZER_DATEI = path.join(
  MODELL_ORDNER,
  "tokenizer.json"
);

const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const GITHUB_OWNER = process.env.GITHUB_OWNER;
const GITHUB_REPO = process.env.GITHUB_REPO;
const GITHUB_BRANCH = process.env.GITHUB_BRANCH || "main";
const ADMIN_KEY = process.env.ADMIN_KEY;

fs.mkdirSync(DATEN_ORDNER, { recursive: true });
fs.mkdirSync(MODELL_ORDNER, { recursive: true });

function jsonAntwort(res, status, daten) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store"
  });

  res.end(JSON.stringify(daten));
}

function leseJsonAnfrage(req, maxBytes = 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let body = "";
    let bytes = 0;

    req.setEncoding("utf8");

    req.on("data", teil => {
      bytes += Buffer.byteLength(teil, "utf8");

      if (bytes > maxBytes) {
        reject(new Error("Anfrage ist zu groß."));
        req.destroy();
        return;
      }

      body += teil;
    });

    req.on("end", () => {
      try {
        resolve(JSON.parse(body || "{}"));
      } catch {
        reject(new Error("Ungültiges JSON."));
      }
    });

    req.on("error", reject);
  });
}

// --------------------------------------------------
// JSON-Dateien aus Daten/ laden
// --------------------------------------------------

const trainingsdaten = [];
const textsammlung = [];

function sammleTextwerte(wert, ergebnis) {
  if (typeof wert === "string") {
    const text = wert.trim();

    if (text.length > 0) {
      ergebnis.push(text);
    }

    return;
  }

  if (Array.isArray(wert)) {
    for (const element of wert) {
      sammleTextwerte(element, ergebnis);
    }

    return;
  }

  if (wert && typeof wert === "object") {
    for (const element of Object.values(wert)) {
      sammleTextwerte(element, ergebnis);
    }
  }
}

function sammleTrainingsbeispiele(wert) {
  if (Array.isArray(wert)) {
    for (const element of wert) {
      sammleTrainingsbeispiele(element);
    }

    return;
  }

  if (!wert || typeof wert !== "object") {
    return;
  }

  if (
    Number.isFinite(wert.x1) &&
    Number.isFinite(wert.x2) &&
    Number.isFinite(wert.target) &&
    [0, 1].includes(wert.x1) &&
    [0, 1].includes(wert.x2) &&
    [0, 1].includes(wert.target)
  ) {
    trainingsdaten.push({
      x1: wert.x1,
      x2: wert.x2,
      target: wert.target
    });

    return;
  }

  for (const element of Object.values(wert)) {
    sammleTrainingsbeispiele(element);
  }
}

function ladeDaten() {
  const dateien = fs.readdirSync(DATEN_ORDNER)
    .filter(name => name.toLowerCase().endsWith(".json"))
    .sort();

  for (const datei of dateien) {
    const dateipfad = path.join(DATEN_ORDNER, datei);

    try {
      const inhalt = fs.readFileSync(dateipfad, "utf8");
      const daten = JSON.parse(inhalt);

      sammleTrainingsbeispiele(daten);
      sammleTextwerte(daten, textsammlung);

      console.log("JSON geladen:", datei);
    } catch (fehler) {
      console.error(
        "JSON-Datei übersprungen:",
        datei,
        fehler.message
      );
    }
  }

  console.log("Trainingsbeispiele:", trainingsdaten.length);
  console.log("Gefundene Textabschnitte:", textsammlung.length);
}

ladeDaten();

// --------------------------------------------------
// Neuronales Netz initialisieren und trainieren
// --------------------------------------------------

let netz = null;
let netzBereit = false;
let netzFehler = null;

try {
  netz = new NeuronalesNetz();
} catch (fehler) {
  netzFehler = fehler.message;
  console.error("Netz konnte nicht initialisiert werden:", fehler.message);
}

function findeMethode(objekt, namen) {
  if (!objekt) return null;

  for (const name of namen) {
    if (typeof objekt[name] === "function") {
      return objekt[name].bind(objekt);
    }
  }

  return null;
}

function trainiereNetz() {
  if (!netz || trainingsdaten.length === 0) {
    return;
  }

  try {
    const batchTraining = findeMethode(netz, [
      "trainiereDatensatz",
      "trainiereDaten",
      "trainBatch",
      "trainOnData"
    ]);

    if (batchTraining) {
      batchTraining(trainingsdaten, 20000);
      netzBereit = true;
      return;
    }

    const einzelTraining = findeMethode(netz, [
      "trainiere",
      "lerne",
      "lernen",
      "train",
      "trainExample"
    ]);

    if (!einzelTraining) {
      throw new Error(
        "Keine passende Trainingsmethode in netz.js gefunden."
      );
    }

    for (let epoche = 0; epoche < 20000; epoche++) {
      for (const beispiel of trainingsdaten) {
        einzelTraining(
          [beispiel.x1, beispiel.x2],
          beispiel.target
        );
      }
    }

    netzBereit = true;
    console.log("Netztraining abgeschlossen.");
  } catch (fehler) {
    netzFehler = fehler.message;
    console.error("Netztraining fehlgeschlagen:", fehler.message);
  }
}

trainiereNetz();

function netzVorhersage(eingaben) {
  if (!netz || !netzBereit) {
    throw new Error(
      netzFehler || "Das neuronale Netz ist noch nicht bereit."
    );
  }

  const vorhersage = findeMethode(netz, [
    "vorhersage",
    "predict",
    "vorwaerts",
    "forward",
    "berechne"
  ]);

  if (!vorhersage) {
    throw new Error(
      "Keine passende Vorhersagemethode in netz.js gefunden."
    );
  }

  let ergebnis = vorhersage(eingaben);

  if (Array.isArray(ergebnis) || ArrayBuffer.isView(ergebnis)) {
    ergebnis = ergebnis[0];
  }

  if (ergebnis && typeof ergebnis === "object") {
    ergebnis =
      ergebnis.ausgabe ??
      ergebnis.output ??
      ergebnis.wert;
  }

  if (typeof ergebnis !== "number" || !Number.isFinite(ergebnis)) {
    throw new Error("Das Netzwerk lieferte kein gültiges Ergebnis.");
  }

  return {
    rohwert: ergebnis,
    vorhersage: ergebnis >= 0.5 ? 1 : 0
  };
}

// --------------------------------------------------
// Tokenizer laden und aus Texten erweitern
// --------------------------------------------------

let tokenizer;

try {
  tokenizer = fs.existsSync(TOKENIZER_DATEI)
    ? Tokenizer.laden(TOKENIZER_DATEI)
    : new Tokenizer();
} catch (fehler) {
  console.error("Tokenizer wird neu erstellt:", fehler.message);
  tokenizer = new Tokenizer();
}

function baueTokenizerAusDaten() {
  if (textsammlung.length === 0) {
    console.log(
      "Keine Textdaten gefunden. " +
      "Für ein Textvokabular brauchst du Texte in Daten/."
    );
    return;
  }

  tokenizer.lerneTexte(textsammlung);
  tokenizer.speichern(TOKENIZER_DATEI);

  console.log(
    "Tokenizer erstellt. Anzahl Tokens:",
    tokenizer.status().anzahlTokens
  );
}

// --------------------------------------------------
// GitHub-API: Datei erstellen oder aktualisieren
// --------------------------------------------------

function githubKonfiguriert() {
  return Boolean(
    GITHUB_TOKEN &&
    GITHUB_OWNER &&
    GITHUB_REPO
  );
}

async function githubDateiSpeichern(dateiInhalt) {
  if (!githubKonfiguriert()) {
    return {
      gespeichert: false,
      meldung: "GitHub-Umgebungsvariablen fehlen."
    };
  }

  const dateipfad = "modelle/tokenizer.json";

  const apiUrl =
    `https://api.github.com/repos/` +
    `${encodeURIComponent(GITHUB_OWNER)}/` +
    `${encodeURIComponent(GITHUB_REPO)}/contents/` +
    dateipfad.split("/").map(encodeURIComponent).join("/");

  const headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": `Bearer ${GITHUB_TOKEN}`,
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "MeineEigeneKI"
  };

  // Prüfen, ob die Datei schon existiert.
  const getAntwort = await fetch(
    `${apiUrl}?ref=${encodeURIComponent(GITHUB_BRANCH)}`,
    { headers }
  );

  let sha = null;

  if (getAntwort.status === 200) {
    const bestehendeDatei = await getAntwort.json();
    sha = bestehendeDatei.sha;

    if (
      bestehendeDatei.encoding === "base64" &&
      typeof bestehendeDatei.content === "string"
    ) {
      const bisherigerInhalt = Buffer.from(
        bestehendeDatei.content.replace(/\s/g, ""),
        "base64"
      ).toString("utf8");

      if (bisherigerInhalt === dateiInhalt) {
        return {
          gespeichert: true,
          unveraendert: true,
          meldung: "Die GitHub-Datei ist bereits aktuell."
        };
      }
    }
  } else if (getAntwort.status !== 404) {
    throw new Error(
      `GitHub-Abfrage fehlgeschlagen: HTTP ${getAntwort.status}`
    );
  }

  const payload = {
    message: "Tokenizer aktualisieren",
    content: Buffer.from(dateiInhalt, "utf8").toString("base64"),
    branch: GITHUB_BRANCH
  };

  // SHA ist beim Ersetzen einer existierenden Datei notwendig.
  if (sha) {
    payload.sha = sha;
  }

  const putAntwort = await fetch(apiUrl, {
    method: "PUT",
    headers: {
      ...headers,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  const putErgebnis = await putAntwort.json().catch(() => ({}));

  if (!putAntwort.ok) {
    console.error("GitHub meldet einen Fehler:", putAntwort.status);

    throw new Error(
      `GitHub konnte die Datei nicht speichern (HTTP ${putAntwort.status}). ` +
      "Prüfe Token, Repository, Branch und Berechtigungen."
    );
  }

  return {
    gespeichert: true,
    url: putErgebnis.content?.html_url || null,
    meldung: "Tokenizer wurde auf GitHub gespeichert."
  };
}

async function tokenizerAufGitHubSynchronisieren() {
  if (!githubKonfiguriert()) {
    console.log(
      "GitHub-Upload übersprungen: Umgebungsvariablen fehlen."
    );
    return;
  }

  try {
    const inhalt = fs.readFileSync(TOKENIZER_DATEI, "utf8");
    const ergebnis = await githubDateiSpeichern(inhalt);

    console.log("GitHub-Tokenizer:", ergebnis.meldung);
    if (ergebnis.url) console.log(ergebnis.url);
  } catch (fehler) {
    console.error("GitHub-Upload fehlgeschlagen:", fehler.message);
  }
}

// Erst lokal erstellen, dann automatisch nach GitHub hochladen.
baueTokenizerAusDaten();

if (textsammlung.length > 0) {
  tokenizerAufGitHubSynchronisieren();
} else {
  // Lokale Datei erstellen, auch wenn noch keine Texte vorhanden sind.
  tokenizer.speichern(TOKENIZER_DATEI);
}

// --------------------------------------------------
// Sicherheit für den Aktualisierungs-Endpunkt
// --------------------------------------------------

function adminKeyIstGueltig(req) {
  const erwartet = process.env.ADMIN_KEY;
  const erhalten = req.headers["x-admin-key"];

  if (
    typeof erwartet !== "string" ||
    erwartet.length === 0 ||
    typeof erhalten !== "string"
  ) {
    return false;
  }

  const a = Buffer.from(erwartet);
  const b = Buffer.from(erhalten);

  return (
    a.length === b.length &&
    crypto.timingSafeEqual(a, b)
  );
}

// --------------------------------------------------
// HTTP-API und Website
// --------------------------------------------------

const server = http.createServer(async (req, res) => {
  const url = new URL(
    req.url,
    `http://${req.headers.host || "localhost"}`
  );

  try {
    if (req.method === "GET" && url.pathname === "/") {
      if (!fs.existsSync(HTML_DATEI)) {
        return jsonAntwort(res, 404, {
          fehler: "index.html wurde nicht gefunden."
        });
      }

      res.writeHead(200, {
        "Content-Type": "text/html; charset=utf-8"
      });

      return res.end(fs.readFileSync(HTML_DATEI));
    }

    if (req.method === "GET" && url.pathname === "/api/status") {
      return jsonAntwort(res, 200, {
        ok: true,
        trainingsbeispiele: trainingsdaten.length,
        netzBereit,
        netzFehler,
        tokenizer: tokenizer.status(),
        githubKonfiguriert: githubKonfiguriert()
      });
    }

    if (
      req.method === "POST" &&
      url.pathname === "/api/vorhersage"
    ) {
      const daten = await leseJsonAnfrage(req);
      const x1 = Number(daten.x1);
      const x2 = Number(daten.x2);

      if (![0, 1].includes(x1) || ![0, 1].includes(x2)) {
        return jsonAntwort(res, 400, {
          fehler: "x1 und x2 müssen jeweils 0 oder 1 sein."
        });
      }

      const ergebnis = netzVorhersage([x1, x2]);

      return jsonAntwort(res, 200, {
        x1,
        x2,
        ...ergebnis
      });
    }

    if (req.method === "POST" && url.pathname === "/api/chat") {
      const daten = await leseJsonAnfrage(req);
      const nachricht = String(
        daten.nachricht ?? daten.message ?? daten.text ?? ""
      ).trim();

      if (!nachricht) {
        return jsonAntwort(res, 400, {
          fehler: "Bitte gib eine Nachricht ein."
        });
      }

      const text = nachricht.toLowerCase();
      let antwort;

      if (/^(hallo|hi|hey|guten morgen)\b/.test(text)) {
        antwort = "Hallo! Schön, dass du da bist.";
      } else if (text.includes("dein name")) {
        antwort = "Ich bin deine selbst entwickelte KI.";
      } else if (text.includes("wie geht")) {
        antwort = "Danke der Nachfrage! Ich bin bereit zu lernen.";
      } else if (
        text.includes("was kannst du") ||
        text.includes("hilfe")
      ) {
        antwort =
          "Ich kann momentan einfache Antworten geben und " +
          "einfache Zahlenbeispiele mit meinem neuronalen Netz berechnen.";
      } else {
        antwort =
          "Ich habe deine Nachricht erhalten. Mein Textmodell muss " +
          "noch entwickelt werden, damit ich daraus eigenständig " +
          "Antworten erzeugen kann.";
      }

      return jsonAntwort(res, 200, {
        antwort,
        reply: antwort
      });
    }

    // Neue Texte lernen und die Datei nach GitHub übertragen.
    if (
      req.method === "POST" &&
      url.pathname === "/api/tokenizer/speichern"
    ) {
      if (!adminKeyIstGueltig(req)) {
        return jsonAntwort(res, 403, {
          fehler: "Zugriff verweigert."
        });
      }

      const daten = await leseJsonAnfrage(req);

      if (
        !Array.isArray(daten.texte) ||
        daten.texte.length === 0 ||
        !daten.texte.every(
          text => typeof text === "string" && text.length <= 5000
        )
      ) {
        return jsonAntwort(res, 400, {
          fehler:
            "Sende ein Array mit dem Namen 'texte', " +
            "das kurze Textzeichenfolgen enthält."
        });
      }

      tokenizer.lerneTexte(daten.texte);
      tokenizer.speichern(TOKENIZER_DATEI);

      const inhalt = fs.readFileSync(TOKENIZER_DATEI, "utf8");
      const githubErgebnis = await githubDateiSpeichern(inhalt);

      return jsonAntwort(res, 200, {
        ok: true,
        tokenizer: tokenizer.status(),
        github: githubErgebnis
      });
    }

    return jsonAntwort(res, 404, {
      fehler: "API-Endpunkt nicht gefunden."
    });
  } catch (fehler) {
    console.error("Anfrage fehlgeschlagen:", fehler.message);

    return jsonAntwort(res, 500, {
      fehler: fehler.message
    });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`Server läuft auf Port ${PORT}.`);
});
