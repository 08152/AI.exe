"use strict";

/*
 * MEINE KI – Lokales neuronales Sprachmodell
 * ------------------------------------------
 * Keine API, keine Cloud, keine externen Pakete.
 *
 * Funktionen:
 * - Rekurrentes neuronales Netz (RNN)
 * - Token-für-Token-Textgenerierung
 * - Zeichenbasierte Verarbeitung unbekannter Wörter
 * - N-Gramm-Sprachmodell als zusätzliche Kontextquelle
 * - Frage-Antwort-Training aus JSON
 * - Training mit normalen Texten
 * - Backpropagation Through Time (BPTT)
 * - Gradient Clipping
 * - Temperatur, Top-K-Sampling und Wiederholungsstrafe
 * - Speichern und Laden des trainierten Modells
 *
 * Hinweis:
 * Dieses Modell ist ein Lernprojekt. Seine Antworten können unlogisch
 * sein. Mehr Code allein bedeutet nicht automatisch mehr Intelligenz.
 */

const fs = require("node:fs");
const path = require("node:path");

const VERSION = 3;

const SPECIAL = Object.freeze({
  PAD: "<PAD>",
  UNK: "<UNK>",
  BOS: "<BOS>",
  EOS: "<EOS>",
  USER: "<BENUTZER>",
  AI: "<KI>",
  WORD: "<WORT>",
  END_WORD: "</WORT>"
});

const SPECIAL_LIST = Object.values(SPECIAL);

const REGEX = Object.freeze({
  TOKEN: /<[^>\s]+>|[\p{L}\p{M}\p{N}_]+(?:['’.-][\p{L}\p{M}\p{N}_]+)*|[^\s]/gu,
  WORD: /^[\p{L}\p{M}\p{N}_]+(?:['’.-][\p{L}\p{M}\p{N}_]+)*$/u,
  CHAR: /^<CHAR:([0-9a-f]+)>$/i
});

const QUESTION_FIELDS = [
  "frage",
  "question",
  "prompt",
  "input",
  "user",
  "benutzer"
];

const ANSWER_FIELDS = [
  "antwort",
  "answer",
  "response",
  "completion",
  "output",
  "assistant",
  "ki"
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function makeVector(length, value = 0) {
  return Array.from({ length }, () => value);
}

function makeMatrix(rows, cols, value = 0) {
  return Array.from(
    { length: rows },
    () => makeVector(cols, value)
  );
}

function randomMatrix(rows, cols, scale) {
  return Array.from(
    { length: rows },
    () => Array.from(
      { length: cols },
      () => (Math.random() * 2 - 1) * scale
    )
  );
}

function shuffleInPlace(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }

  return array;
}

function stableSoftmax(logits, temperature = 1) {
  if (!logits.length) return [];

  temperature = clamp(temperature, 0.05, 5);

  let maxLogit = -Infinity;

  for (const value of logits) {
    if (Number.isFinite(value) && value > maxLogit) {
      maxLogit = value;
    }
  }

  if (!Number.isFinite(maxLogit)) {
    return makeVector(logits.length, 1 / logits.length);
  }

  const probabilities = new Array(logits.length);
  let total = 0;

  for (let i = 0; i < logits.length; i++) {
    const value = Number.isFinite(logits[i])
      ? Math.exp(clamp(
          (logits[i] - maxLogit) / temperature,
          -60,
          0
        ))
      : 0;

    probabilities[i] = value;
    total += value;
  }

  if (!Number.isFinite(total) || total <= 0) {
    return makeVector(logits.length, 1 / logits.length);
  }

  for (let i = 0; i < probabilities.length; i++) {
    probabilities[i] /= total;
  }

  return probabilities;
}

function weightedChoice(probabilities) {
  let total = 0;

  for (const probability of probabilities) {
    if (Number.isFinite(probability) && probability > 0) {
      total += probability;
    }
  }

  if (total <= 0) return -1;

  let random = Math.random() * total;

  for (let i = 0; i < probabilities.length; i++) {
    const probability = probabilities[i];

    if (!Number.isFinite(probability) || probability <= 0) {
      continue;
    }

    random -= probability;

    if (random <= 0) return i;
  }

  for (let i = probabilities.length - 1; i >= 0; i--) {
    if (probabilities[i] > 0) return i;
  }

  return -1;
}

function shuffleCopy(array) {
  return shuffleInPlace(array.slice());
}

function safeMean(values) {
  if (!values.length) return 0;

  let sum = 0;

  for (const value of values) {
    if (Number.isFinite(value)) sum += value;
  }

  return sum / values.length;
}

class NeuronalesNetz {
  constructor(tokenizer = null, optionen = {}) {
    /*
     * Unterstützt beide Aufrufe:
     * new NeuronalesNetz()
     * new NeuronalesNetz(null, { versteckteNeuronen: 64 })
     */
    if (
      tokenizer &&
      typeof tokenizer.zerlege !== "function"
    ) {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.tokenizer = tokenizer || null;

    this.optionen = {
      maxVokabular: 1200,
      embeddingGroesse: 32,
      versteckteNeuronen: 64,
      kontextLaenge: 48,
      segmentLaenge: 16,
      maxTrainingsTokens: 12000,
      ngramOrdnung: 5,
      lernrate: 0.025,
      epochen: 8,
      gradientGrenze: 3,
      maxWortZeichen: 48
    };

    for (const [key, value] of Object.entries(optionen || {})) {
      if (key in this.optionen && isFiniteNumber(value)) {
        this.optionen[key] = value;
      }
    }

    this.optionen.maxVokabular = Math.max(
      100,
      Math.floor(this.optionen.maxVokabular)
    );

    this.optionen.embeddingGroesse = Math.max(
      8,
      Math.floor(this.optionen.embeddingGroesse)
    );

    this.optionen.versteckteNeuronen = Math.max(
      16,
      Math.floor(this.optionen.versteckteNeuronen)
    );

    this.optionen.kontextLaenge = Math.max(
      4,
      Math.floor(this.optionen.kontextLaenge)
    );

    this.optionen.segmentLaenge = clamp(
      Math.floor(this.optionen.segmentLaenge),
      4,
      64
    );

    this.optionen.ngramOrdnung = clamp(
      Math.floor(this.optionen.ngramOrdnung),
      2,
      7
    );

    this.maxVokabular = this.optionen.maxVokabular;
    this.embeddingGroesse = this.optionen.embeddingGroesse;
    this.versteckteNeuronen = this.optionen.versteckteNeuronen;
    this.kontextLaenge = this.optionen.kontextLaenge;
    this.trainingsSchrittLaenge = this.optionen.segmentLaenge;
    this.maxTrainingsTokens = this.optionen.maxTrainingsTokens;
    this.ngramOrdnung = this.optionen.ngramOrdnung;

    this.vokabular = [];
    this.tokenZuId = new Map();

    this.embeddings = [];
    this.gewichteEingabe = [];
    this.gewichteRekurrenz = [];
    this.biasVersteckt = [];
    this.gewichteAusgabe = [];
    this.biasAusgabe = [];

    this.ngramCounts = [];
    this.wortHaeufigkeit = new Map();
    this.haeufigeWoerter = new Set();

    this.trainingsSequenzen = [];
    this.trainingsPaare = [];
    this.trainingsBeispiele = 0;
    this.trainierteEpochen = 0;

    this.verlustHistorie = [];
    this.letzteAntworten = [];
    this.letzteAntwortAnalyse = null;
    this.letzterFehler = null;

    this.bereit = false;
    this.konversationsModus = false;

    this.padId = 0;
    this.unkId = 1;
    this.bosId = 2;
    this.eosId = 3;
    this.userId = 4;
    this.aiId = 5;
    this.wordId = 6;
    this.endWordId = 7;
  }

  normalisiereText(text) {
    return String(text ?? "")
      .normalize("NFC")
      .replace(/\r\n?/g, "\n")
      .trim();
  }

  zerlegeText(text) {
    const sauber = this.normalisiereText(text);

    return sauber.match(REGEX.TOKEN) || [];
  }

  charToken(char) {
    return `<CHAR:${char.codePointAt(0).toString(16)}>`;
  }

  charAusToken(token) {
    const match = REGEX.CHAR.exec(token || "");

    if (!match) return null;

    try {
      return String.fromCodePoint(
        parseInt(match[1], 16)
      );
    } catch {
      return null;
    }
  }

  istWortToken(token) {
    return REGEX.WORD.test(token);
  }

  findeFeld(objekt, namen) {
    const keys = new Map(
      Object.keys(objekt).map(key => [
        key.toLocaleLowerCase("de-DE"),
        key
      ])
    );

    for (const name of namen) {
      const key = keys.get(name);

      if (key !== undefined) return key;
    }

    return null;
  }

  sammleTrainingsDaten(daten) {
    const sequenzen = [];
    const paare = [];

    const besuchen = wert => {
      if (typeof wert === "string") {
        const text = this.normalisiereText(wert);

        if (text) {
          sequenzen.push({
            art: "text",
            text
          });
        }

        return;
      }

      if (Array.isArray(wert)) {
        for (const element of wert) {
          besuchen(element);
        }

        return;
      }

      if (!wert || typeof wert !== "object") {
        return;
      }

      const frageFeld = this.findeFeld(
        wert,
        QUESTION_FIELDS
      );

      const antwortFeld = this.findeFeld(
        wert,
        ANSWER_FIELDS
      );

      if (
        frageFeld &&
        antwortFeld &&
        typeof wert[frageFeld] === "string" &&
        typeof wert[antwortFeld] === "string"
      ) {
        const frage = this.normalisiereText(
          wert[frageFeld]
        );

        const antwort = this.normalisiereText(
          wert[antwortFeld]
        );

        if (frage && antwort) {
          const paar = { frage, antwort };

          paare.push(paar);

          sequenzen.push({
            art: "dialog",
            frage,
            antwort
          });
        }

        return;
      }

      for (const unterwert of Object.values(wert)) {
        besuchen(unterwert);
      }
    };

    besuchen(daten);

    return {
      sequenzen,
      paare
    };
  }

  tokensDerSequenz(sequenz) {
    if (sequenz.art === "dialog") {
      return [
        SPECIAL.BOS,
        SPECIAL.USER,
        ...this.zerlegeText(sequenz.frage),
        SPECIAL.AI,
        ...this.zerlegeText(sequenz.antwort),
        SPECIAL.EOS
      ];
    }

    return [
      SPECIAL.BOS,
      ...this.zerlegeText(sequenz.text),
      SPECIAL.EOS
    ];
  }

  baueVokabular(sequenzen) {
    const haeufigkeiten = new Map();
    const zeichen = new Set();
    const sonderTokens = new Set();

    for (const sequenz of sequenzen) {
      for (const token of this.tokensDerSequenz(sequenz)) {
        if (SPECIAL_LIST.includes(token)) continue;

        haeufigkeiten.set(
          token,
          (haeufigkeiten.get(token) || 0) + 1
        );

        if (this.istWortToken(token)) {
          for (const char of Array.from(token)) {
            zeichen.add(char);
          }
        } else {
          sonderTokens.add(token);
        }
      }
    }

    const woerter = [...haeufigkeiten.entries()]
      .filter(([token]) => this.istWortToken(token))
      .sort((a, b) => b[1] - a[1])
      .slice(0, this.maxVokabular)
      .map(([token]) => token);

    this.haeufigeWoerter = new Set(woerter);
    this.wortHaeufigkeit = haeufigkeiten;

    const zeichenTokens = [...zeichen].map(
      char => this.charToken(char)
    );

    this.vokabular = [...new Set([
      ...SPECIAL_LIST,
      ...woerter,
      ...sonderTokens,
      ...zeichenTokens
    ])];

    this.tokenZuId = new Map(
      this.vokabular.map((token, id) => [token, id])
    );

    this.padId = this.tokenZuId.get(SPECIAL.PAD);
    this.unkId = this.tokenZuId.get(SPECIAL.UNK);
    this.bosId = this.tokenZuId.get(SPECIAL.BOS);
    this.eosId = this.tokenZuId.get(SPECIAL.EOS);
    this.userId = this.tokenZuId.get(SPECIAL.USER);
    this.aiId = this.tokenZuId.get(SPECIAL.AI);
    this.wordId = this.tokenZuId.get(SPECIAL.WORD);
    this.endWordId = this.tokenZuId.get(SPECIAL.END_WORD);
  }

  kodiereEinToken(token) {
    const direkt = this.tokenZuId.get(token);

    if (direkt !== undefined) return [direkt];

    if (this.istWortToken(token)) {
      const ids = [this.wordId];

      for (const char of Array.from(token)) {
        const id = this.tokenZuId.get(
          this.charToken(char)
        );

        ids.push(id === undefined ? this.unkId : id);
      }

      ids.push(this.endWordId);

      return ids;
    }

    return [this.unkId];
  }

  kodiereText(text, mitBOS = false, mitEOS = false) {
    const ids = [];

    if (mitBOS) ids.push(this.bosId);

    for (const token of this.zerlegeText(text)) {
      ids.push(...this.kodiereEinToken(token));
    }

    if (mitEOS) ids.push(this.eosId);

    return ids;
  }

  kodiereSequenz(sequenz) {
    const ids = [];

    for (const token of this.tokensDerSequenz(sequenz)) {
      ids.push(...this.kodiereEinToken(token));
    }

    return ids;
  }

  initialisiereGewichte() {
    const V = this.vokabular.length;
    const E = this.embeddingGroesse;
    const H = this.versteckteNeuronen;

    this.embeddings = randomMatrix(
      V,
      E,
      Math.sqrt(1 / E)
    );

    this.gewichteEingabe = randomMatrix(
      E,
      H,
      Math.sqrt(2 / (E + H))
    );

    this.gewichteRekurrenz = randomMatrix(
      H,
      H,
      Math.sqrt(1 / H)
    );

    this.biasVersteckt = makeVector(H);

    this.gewichteAusgabe = randomMatrix(
      H,
      V,
      Math.sqrt(2 / (H + V))
    );

    this.biasAusgabe = makeVector(V);
  }
    berechneVorwaerts(tokenIds, trainingsModus = false) {
    const H = this.versteckteNeuronen;
    const E = this.embeddingGroesse;
    const V = this.vokabular.length;

    let zustand = makeVector(H, 0);
    const schritte = [];

    for (let t = 0; t < tokenIds.length; t++) {
      const tokenId = clamp(tokenIds[t] ?? this.unkId, 0, V - 1);
      const embedding = this.embeddings[tokenId];
      const vorherigerZustand = zustand.slice();

      const vorAktivierung = makeVector(H, 0);

      for (let h = 0; h < H; h++) {
        let summe = this.biasVersteckt[h];

        for (let e = 0; e < E; e++) {
          summe += this.gewichteEingabe[h][e] * embedding[e];
        }

        for (let j = 0; j < H; j++) {
          summe += this.gewichteRekurrenz[h][j] * vorherigerZustand[j];
        }

        vorAktivierung[h] = summe;
      }

      zustand = vorAktivierung.map(x => Math.tanh(clamp(x, -20, 20)));

      const logits = makeVector(V, 0);

      for (let v = 0; v < V; v++) {
        let summe = this.biasAusgabe[v];

        for (let h = 0; h < H; h++) {
          summe += this.gewichteAusgabe[v][h] * zustand[h];
        }

        logits[v] = clamp(summe, -30, 30);
      }

      const wahrscheinlichkeiten = stableSoftmax(logits);

      if (trainingsModus) {
        schritte.push({
          tokenId,
          embedding: embedding.slice(),
          vorherigerZustand,
          zustand: zustand.slice(),
          logits,
          wahrscheinlichkeiten
        });
      }
    }

    return {
      zustand,
      schritte,
      logits: this.vorhersageLogits(zustand),
      wahrscheinlichkeiten: stableSoftmax(this.vorhersageLogits(zustand))
    };
  }

  vorhersageLogits(zustand) {
    const V = this.vokabular.length;
    const H = this.versteckteNeuronen;
    const logits = makeVector(V, 0);

    for (let v = 0; v < V; v++) {
      let summe = this.biasAusgabe[v];

      for (let h = 0; h < H; h++) {
        summe += this.gewichteAusgabe[v][h] * zustand[h];
      }

      logits[v] = clamp(summe, -30, 30);
    }

    return logits;
  }

  berechneVerlust(wahrscheinlichkeiten, zielId) {
    const p = clamp(wahrscheinlichkeiten[zielId] ?? 1e-9, 1e-9, 1);
    return -Math.log(p);
  }

  baueNGramme(tokens, ordnung = this.ngramOrdnung) {
    const ergebnis = new Map();

    for (let i = 0; i < tokens.length; i++) {
      const ziel = tokens[i];

      for (let n = 1; n <= ordnung; n++) {
        const start = Math.max(0, i - n);
        const kontext = tokens.slice(start, i);
        const schluessel = kontext.join(" ") + " => " + ziel;

        ergebnis.set(
          schluessel,
          (ergebnis.get(schluessel) || 0) + 1
        );
      }
    }

    return ergebnis;
  }

  aktualisiereNGramme(tokens) {
    for (let i = 0; i < tokens.length; i++) {
      const ziel = tokens[i];

      for (let n = 1; n <= this.ngramOrdnung; n++) {
        const start = Math.max(0, i - n);
        const kontext = tokens.slice(start, i);
        const kontextSchluessel = kontext.join(" ");

        if (!this.ngramCounts.has(kontextSchluessel)) {
          this.ngramCounts.set(kontextSchluessel, new Map());
        }

        const ziele = this.ngramCounts.get(kontextSchluessel);
        ziele.set(ziel, (ziele.get(ziel) || 0) + 1);
      }
    }
  }

  findeNGramVorhersage(tokens) {
    for (
      let n = Math.min(this.ngramOrdnung, tokens.length);
      n >= 0;
      n--
    ) {
      const kontext = tokens.slice(-n).join(" ");
      const ziele = this.ngramCounts.get(kontext);

      if (!ziele || ziele.size === 0) continue;

      let gesamt = 0;

      for (const anzahl of ziele.values()) {
        gesamt += anzahl;
      }

      const sortiert = Array.from(ziele.entries())
        .sort((a, b) => b[1] - a[1]);

      return {
        kontext,
        gesamt,
        kandidaten: sortiert.map(([token, anzahl]) => ({
          token,
          anzahl,
          anteil: anzahl / Math.max(1, gesamt)
        }))
      };
    }

    return null;
  }

  berechneGradienten(schritte, zielIds) {
    const H = this.versteckteNeuronen;
    const E = this.embeddingGroesse;
    const V = this.vokabular.length;

    const grad = {
      embeddings: makeMatrix(V, E),
      gewichteEingabe: makeMatrix(H, E),
      gewichteRekurrenz: makeMatrix(H, H),
      biasVersteckt: makeVector(H),
      gewichteAusgabe: makeMatrix(V, H),
      biasAusgabe: makeVector(V),
      verlust: 0,
      anzahl: 0
    };

    if (!schritte.length || !zielIds.length) return grad;

    const lernrate = this.optionen.lernrate;
    const gradientGrenze = this.optionen.gradientGrenze;

    for (let t = 0; t < schritte.length; t++) {
      const schritt = schritte[t];
      const zielId = zielIds[t];

      if (zielId == null || zielId < 0 || zielId >= V) continue;

      const probs = schritt.wahrscheinlichkeiten;
      grad.verlust += this.berechneVerlust(probs, zielId);
      grad.anzahl++;

      const deltaAusgabe = probs.slice();
      deltaAusgabe[zielId] -= 1;

      for (let v = 0; v < V; v++) {
        const delta = deltaAusgabe[v];

        grad.biasAusgabe[v] += delta;

        for (let h = 0; h < H; h++) {
          grad.gewichteAusgabe[v][h] +=
            delta * schritt.zustand[h];
        }
      }

      const deltaVersteckt = makeVector(H, 0);

      for (let h = 0; h < H; h++) {
        let summe = 0;

        for (let v = 0; v < V; v++) {
          summe +=
            this.gewichteAusgabe[v][h] *
            deltaAusgabe[v];
        }

        const aktivierung = schritt.zustand[h];
        deltaVersteckt[h] =
          summe * (1 - aktivierung * aktivierung);
      }

      for (let h = 0; h < H; h++) {
        const delta = deltaVersteckt[h];

        grad.biasVersteckt[h] += delta;

        for (let e = 0; e < E; e++) {
          grad.gewichteEingabe[h][e] +=
            delta * schritt.embedding[e];

          grad.embeddings[schritt.tokenId][e] +=
            delta * this.gewichteEingabe[h][e];
        }

        for (let j = 0; j < H; j++) {
          grad.gewichteRekurrenz[h][j] +=
            delta * schritt.vorherigerZustand[j];
        }
      }
    }

    if (grad.anzahl > 0) {
      const faktor = lernrate / grad.anzahl;

      for (const matrixName of [
        "embeddings",
        "gewichteEingabe",
        "gewichteRekurrenz",
        "gewichteAusgabe"
      ]) {
        const matrix = grad[matrixName];

        for (let i = 0; i < matrix.length; i++) {
          for (let j = 0; j < matrix[i].length; j++) {
            matrix[i][j] = clamp(
              matrix[i][j] * faktor,
              -gradientGrenze,
              gradientGrenze
            );
          }
        }
      }

      for (const vectorName of [
        "biasVersteckt",
        "biasAusgabe"
      ]) {
        grad[vectorName] = grad[vectorName].map(x =>
          clamp(x * faktor, -gradientGrenze, gradientGrenze)
        );
      }
    }

    return grad;
  }

  wendeGradientenAn(grad) {
    const lernrate = this.optionen.lernrate;

    const matrixPaare = [
      ["embeddings", this.embeddings],
      ["gewichteEingabe", this.gewichteEingabe],
      ["gewichteRekurrenz", this.gewichteRekurrenz],
      ["gewichteAusgabe", this.gewichteAusgabe]
    ];

    for (const [name, ziel] of matrixPaare) {
      const quelle = grad[name];

      for (let i = 0; i < ziel.length; i++) {
        for (let j = 0; j < ziel[i].length; j++) {
          ziel[i][j] -= quelle[i][j];
          ziel[i][j] = clamp(ziel[i][j], -8, 8);
        }
      }
    }

    for (const name of ["biasVersteckt", "biasAusgabe"]) {
      const ziel = this[name];
      const quelle = grad[name];

      for (let i = 0; i < ziel.length; i++) {
        ziel[i] -= quelle[i];
        ziel[i] = clamp(ziel[i], -8, 8);
      }
    }
  }

  trainiereEinBeispiel(tokenIds) {
    if (!Array.isArray(tokenIds) || tokenIds.length < 2) {
      return { verlust: 0, tokens: 0 };
    }

    const maxTokens = Math.min(
      tokenIds.length,
      this.maxTrainingsTokens
    );

    let gesamtVerlust = 0;
    let anzahlTokens = 0;

    for (
      let start = 0;
      start < maxTokens - 1;
      start += this.trainingsSchrittLaenge
    ) {
      const ende = Math.min(
        maxTokens,
        start + this.trainingsSchrittLaenge + 1
      );

      const eingabe = tokenIds.slice(start, ende - 1);
      const ziele = tokenIds.slice(start + 1, ende);

      if (!eingabe.length || !ziele.length) continue;

      const vorwaerts = this.berechneVorwaerts(
        eingabe,
        true
      );

      const grad = this.berechneGradienten(
        vorwaerts.schritte,
        ziele
      );

      this.wendeGradientenAn(grad);

      gesamtVerlust += grad.verlust;
      anzahlTokens += grad.anzahl;
    }

    return {
      verlust: anzahlTokens
        ? gesamtVerlust / anzahlTokens
        : 0,
      tokens: anzahlTokens
    };
  }

  trainiereNGramModelle() {
    this.ngramCounts.clear();

    for (const sequenz of this.trainingsSequenzen) {
      const tokens = sequenz.map(id =>
        this.vokabular[id] ?? SPECIAL.UNK
      );

      this.aktualisiereNGramme(tokens);
    }
  }

  trainiereEpoche() {
    const daten = shuffleCopy(this.trainingsSequenzen);
    let verlustSumme = 0;
    let tokenSumme = 0;

    for (const sequenz of daten) {
      const ergebnis = this.trainiereEinBeispiel(sequenz);

      verlustSumme += ergebnis.verlust * ergebnis.tokens;
      tokenSumme += ergebnis.tokens;
    }

    return {
      verlust: tokenSumme
        ? verlustSumme / tokenSumme
        : 0,
      tokens: tokenSumme
    };
  }
    trainiere(daten, optionen = {}) {
    try {
      this.letzterFehler = null;

      if (!Array.isArray(daten) || daten.length === 0) {
        throw new Error("Keine Trainingsdaten vorhanden.");
      }

      const epochen = Math.max(
        1,
        Math.floor(optionen.epochen ?? this.optionen.epochen)
      );

      const rohDaten = this.sammleTrainingsDaten(daten);

      if (!rohDaten.length) {
        throw new Error(
          "Keine gültigen Frage-Antwort-Beispiele gefunden."
        );
      }

      this.baueVokabular(rohDaten);
      this.initialisiereGewichte();

      this.trainingsPaare = rohDaten;
      this.trainingsSequenzen = [];

      for (const beispiel of rohDaten) {
        const tokens = [
          SPECIAL.BOS,
          SPECIAL.USER,
          ...this.zerlegeText(beispiel.frage),
          SPECIAL.AI,
          ...this.zerlegeText(beispiel.antwort),
          SPECIAL.EOS
        ];

        const ids = this.kodiereSequenz(tokens);

        if (ids.length >= 2) {
          this.trainingsSequenzen.push(ids);
        }
      }

      if (!this.trainingsSequenzen.length) {
        throw new Error("Die Trainingssequenzen sind leer.");
      }

      this.trainingsBeispiele = rohDaten.length;
      this.trainierteEpochen = 0;
      this.verlustHistorie = [];

      this.trainiereNGramModelle();

      for (let epoche = 0; epoche < epochen; epoche++) {
        const ergebnis = this.trainiereEpoche();

        this.trainierteEpochen++;

        this.verlustHistorie.push({
          epoche: this.trainierteEpochen,
          verlust: ergebnis.verlust,
          tokens: ergebnis.tokens
        });
      }

      this.bereit = true;

      return {
        erfolg: true,
        beispiele: this.trainingsBeispiele,
        vokabularGroesse: this.vokabular.length,
        epochen: this.trainierteEpochen,
        verlust: this.verlustHistorie.at(-1)?.verlust ?? 0
      };
    } catch (fehler) {
      this.letzterFehler = fehler.message;

      return {
        erfolg: false,
        fehler: fehler.message
      };
    }
  }

  waehleToken(wahrscheinlichkeiten, bisherigeIds = [], optionen = {}) {
    const temperatur = clamp(optionen.temperatur ?? 0.85, 0.1, 2);
    const topK = Math.max(
      1,
      Math.floor(optionen.topK ?? 12)
    );

    const verboten = new Set([
      this.padId,
      this.bosId,
      this.userId
    ]);

    const kandidaten = [];

    for (let id = 0; id < wahrscheinlichkeiten.length; id++) {
      if (verboten.has(id)) continue;

      const token = this.vokabular[id];

      if (!token || token === SPECIAL.AI) continue;

      let p = wahrscheinlichkeiten[id];

      if (!Number.isFinite(p) || p <= 0) continue;

      if (bisherigeIds.length >= 3) {
        const letzteDrei = bisherigeIds.slice(-3);

        if (letzteDrei.every(x => x === id)) {
          p *= 0.05;
        }
      }

      p = Math.pow(p, 1 / temperatur);

      kandidaten.push({ id, p });
    }

    kandidaten.sort((a, b) => b.p - a.p);

    const top = kandidaten.slice(0, topK);

    if (!top.length) return this.unkId;

    const summe = top.reduce((s, k) => s + k.p, 0);

    if (summe <= 0) return top[0].id;

    let zufall = Math.random() * summe;

    for (const kandidat of top) {
      zufall -= kandidat.p;

      if (zufall <= 0) return kandidat.id;
    }

    return top[0].id;
  }

  generiereMitNetz(prompt, optionen = {}) {
    const promptTokens = this.zerlegeText(prompt);
    const ids = this.kodiereSequenz([
      SPECIAL.BOS,
      SPECIAL.USER,
      ...promptTokens,
      SPECIAL.AI
    ]);

    const maxNeu = Math.max(
      1,
      Math.min(160, Math.floor(optionen.maxTokens ?? 60))
    );

    const temperatur = clamp(optionen.temperatur ?? 0.85, 0.1, 2);
    const generierteIds = [];
    let kontext = ids.slice(-this.kontextLaenge);

    for (let i = 0; i < maxNeu; i++) {
      const vorwaerts = this.berechneVorwaerts(kontext);

      const nextId = this.waehleToken(
        stableSoftmax(
          vorwaerts.logits.map(x => x / temperatur)
        ),
        generierteIds,
        {
          temperatur,
          topK: optionen.topK ?? 12
        }
      );

      if (
        nextId === this.eosId ||
        nextId === this.padId ||
        nextId === this.bosId
      ) {
        break;
      }

      generierteIds.push(nextId);
      kontext = [...kontext, nextId].slice(-this.kontextLaenge);
    }

    return this.dekodiereIds(generierteIds);
  }

  dekodiereIds(ids) {
    const tokens = [];

    for (const id of ids) {
      const token = this.vokabular[id];

      if (!token) continue;

      if (token.startsWith("<CHAR:")) {
        const zeichen = this.charAusToken(token);

        if (zeichen) tokens.push(zeichen);
        continue;
      }

      if (token === SPECIAL.EOS) break;

      if (SPECIAL_LIST.includes(token)) continue;

      tokens.push(token);
    }

    let text = "";

    for (const token of tokens) {
      if (!text) {
        text = token;
      } else if (/^[.,!?;:%)\]}…]/u.test(token)) {
        text += token;
      } else if (/^[([{„“"']/u.test(token)) {
        text += " " + token;
      } else {
        text += " " + token;
      }
    }

    return text.trim();
  }

  antworte(frage, optionen = {}) {
    try {
      if (!this.bereit) {
        return {
          text: "Ich bin noch nicht trainiert. Bitte lade Trainingsdaten und trainiere mich zuerst.",
          methode: "status"
        };
      }

      const normalisierteFrage = this.normalisiereText(frage);

      if (!normalisierteFrage) {
        return {
          text: "Bitte gib eine Frage oder Nachricht ein.",
          methode: "validierung"
        };
      }

      const frageTokens = this.zerlegeText(normalisierteFrage);

      const ngram = this.findeNGramVorhersage(frageTokens);

      let besteAntwort = null;
      let bestePunktzahl = -Infinity;

      for (const beispiel of this.trainingsPaare) {
        const tokens = this.zerlegeText(beispiel.frage);
        const antwortTokens = this.zerlegeText(beispiel.antwort);

        const frageMenge = new Set(frageTokens);
        const trainingsMenge = new Set(tokens);

        let gemeinsam = 0;

        for (const token of frageMenge) {
          if (trainingsMenge.has(token)) gemeinsam++;
        }

        const union = new Set([...frageMenge, ...trainingsMenge]).size;
        const aehnlichkeit = union ? gemeinsam / union : 0;

        let gleichePositionen = 0;
        const maxVergleich = Math.min(frageTokens.length, tokens.length);

        for (let i = 0; i < maxVergleich; i++) {
          if (frageTokens[i] === tokens[i]) gleichePositionen++;
        }

        const positionsPunktzahl = maxVergleich
          ? gleichePositionen / maxVergleich
          : 0;

        const laengenFaktor =
          1 / (1 + Math.abs(frageTokens.length - tokens.length) * 0.04);

        const punktzahl =
          aehnlichkeit * 0.7 +
          positionsPunktzahl * 0.2 +
          laengenFaktor * 0.1;

        if (punktzahl > bestePunktzahl) {
          bestePunktzahl = punktzahl;
          besteAntwort = beispiel.antwort;
        }
      }

      const mindestAehnlichkeit = optionen.mindestAehnlichkeit ?? 0.12;

      if (
        besteAntwort &&
        bestePunktzahl >= mindestAehnlichkeit &&
        !optionen.freieGenerierung
      ) {
        const ergebnis = {
          text: besteAntwort,
          methode: "beispielvergleich",
          aehnlichkeit: Number(bestePunktzahl.toFixed(4))
        };

        this.letzteAntworten.push({
          frage: normalisierteFrage,
          antwort: ergebnis.text,
          methode: ergebnis.methode,
          zeit: new Date().toISOString()
        });

        if (this.letzteAntworten.length > 100) {
          this.letzteAntworten.shift();
        }

        this.letzteAntwortAnalyse = ergebnis;
        return ergebnis;
      }

      if (optionen.freieGenerierung) {
        const text = this.generiereMitNetz(
          normalisierteFrage,
          optionen
        );

        const ergebnis = {
          text: text || "Ich kann darauf noch keine passende Antwort bilden.",
          methode: "neurales-netz",
          ngramVorhanden: Boolean(ngram)
        };

        this.letzteAntworten.push({
          frage: normalisierteFrage,
          antwort: ergebnis.text,
          methode: ergebnis.methode,
          zeit: new Date().toISOString()
        });

        if (this.letzteAntworten.length > 100) {
          this.letzteAntworten.shift();
        }

        this.letzteAntwortAnalyse = ergebnis;
        return ergebnis;
      }

      const ergebnis = {
        text: "Das weiß ich noch nicht zuverlässig. Trainiere mich mit weiteren Beispielen zu diesem Thema.",
        methode: "fallback",
        aehnlichkeit: Number(Math.max(0, bestePunktzahl).toFixed(4))
      };

      this.letzteAntwortAnalyse = ergebnis;
      return ergebnis;
    } catch (fehler) {
      this.letzterFehler = fehler.message;

      return {
        text: "Beim Erzeugen der Antwort ist ein Fehler aufgetreten.",
        methode: "fehler",
        fehler: fehler.message
      };
    }
  }

  statistik() {
    return {
      version: VERSION,
      bereit: this.bereit,
      vokabularGroesse: this.vokabular.length,
      trainingsBeispiele: this.trainingsBeispiele,
      trainingsSequenzen: this.trainingsSequenzen.length,
      trainierteEpochen: this.trainierteEpochen,
      embeddingGroesse: this.embeddingGroesse,
      versteckteNeuronen: this.versteckteNeuronen,
      kontextLaenge: this.kontextLaenge,
      ngramOrdnung: this.ngramOrdnung,
      anzahlNGramKontexte: this.ngramCounts.size,
      letzterVerlust: this.verlustHistorie.at(-1)?.verlust ?? null,
      letzterFehler: this.letzterFehler
    };
  }

  speichere(dateiPfad) {
    const ziel = path.resolve(dateiPfad);
    const temporaer = ziel + ".tmp";

    const daten = {
      version: VERSION,
      optionen: this.optionen,
      vokabular: this.vokabular,
      embeddings: this.embeddings,
      gewichteEingabe: this.gewichteEingabe,
      gewichteRekurrenz: this.gewichteRekurrenz,
      biasVersteckt: this.biasVersteckt,
      gewichteAusgabe: this.gewichteAusgabe,
      biasAusgabe: this.biasAusgabe,
      trainingsPaare: this.trainingsPaare,
      trainierteEpochen: this.trainierteEpochen,
      verlustHistorie: this.verlustHistorie,
      bereit: this.bereit
    };

    fs.mkdirSync(path.dirname(ziel), { recursive: true });
    fs.writeFileSync(
      temporaer,
      JSON.stringify(daten),
      "utf8"
    );
    fs.renameSync(temporaer, ziel);

    return { erfolg: true, datei: ziel };
  }

  lade(dateiPfad) {
    try {
      const quelle = path.resolve(dateiPfad);
      const daten = JSON.parse(fs.readFileSync(quelle, "utf8"));

      if (!daten || !Array.isArray(daten.vokabular)) {
        throw new Error("Ungültige Modelldatei.");
      }

      if (daten.version !== VERSION) {
        throw new Error(
          "Die Modelldatei hat eine andere Version. Bitte trainiere das Modell neu."
        );
      }

      this.optionen = {
        ...this.optionen,
        ...(daten.optionen || {})
      };

      this.maxVokabular = this.optionen.maxVokabular;
      this.embeddingGroesse = this.optionen.embeddingGroesse;
      this.versteckteNeuronen = this.optionen.versteckteNeuronen;
      this.kontextLaenge = this.optionen.kontextLaenge;
      this.trainingsSchrittLaenge = this.optionen.segmentLaenge;
      this.maxTrainingsTokens = this.optionen.maxTrainingsTokens;
      this.ngramOrdnung = this.optionen.ngramOrdnung;

      this.vokabular = daten.vokabular;
      this.tokenZuId = new Map(
        this.vokabular.map((token, id) => [token, id])
      );

      this.padId = this.tokenZuId.get(SPECIAL.PAD) ?? 0;
      this.unkId = this.tokenZuId.get(SPECIAL.UNK) ?? 1;
      this.bosId = this.tokenZuId.get(SPECIAL.BOS) ?? 2;
      this.eosId = this.tokenZuId.get(SPECIAL.EOS) ?? 3;
      this.userId = this.tokenZuId.get(SPECIAL.USER) ?? 4;
      this.aiId = this.tokenZuId.get(SPECIAL.AI) ?? 5;

      this.embeddings = daten.embeddings;
      this.gewichteEingabe = daten.gewichteEingabe;
      this.gewichteRekurrenz = daten.gewichteRekurrenz;
      this.biasVersteckt = daten.biasVersteckt;
      this.gewichteAusgabe = daten.gewichteAusgabe;
      this.biasAusgabe = daten.biasAusgabe;

      this.trainingsPaare = daten.trainingsPaare || [];
      this.trainingsBeispiele = this.trainingsPaare.length;
      this.trainierteEpochen = daten.trainierteEpochen || 0;
      this.verlustHistorie = daten.verlustHistorie || [];

      this.trainingsSequenzen = this.trainingsPaare.map(beispiel => {
        const tokens = [
          SPECIAL.BOS,
          SPECIAL.USER,
          ...this.zerlegeText(beispiel.frage),
          SPECIAL.AI,
          ...this.zerlegeText(beispiel.antwort),
          SPECIAL.EOS
        ];

        return this.kodiereSequenz(tokens);
      });

      this.ngramCounts.clear();
      this.trainiereNGramModelle();

      this.bereit = Boolean(daten.bereit);

      return {
        erfolg: true,
        statistik: this.statistik()
      };
    } catch (fehler) {
      this.letzterFehler = fehler.message;

      return {
        erfolg: false,
        fehler: fehler.message
      };
    }
  }
}

module.exports = {
  NeuronalesNetz
};
