"use strict";

const fs = require("node:fs");
const path = require("node:path");

const VERSION = 4;

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

const TOKEN_RE =
  /<[^>\s]+>|[\p{L}\p{M}\p{N}_]+(?:['’.-][\p{L}\p{M}\p{N}_]+)*|[^\s]/gu;

const WORD_RE =
  /^[\p{L}\p{M}\p{N}_]+(?:['’.-][\p{L}\p{M}\p{N}_]+)*$/u;

const CHAR_RE = /^<CHAR:([0-9a-f]+)>$/i;

const QUESTION_FIELDS = [
  "frage", "question", "prompt", "input", "user", "benutzer"
];

const ANSWER_FIELDS = [
  "antwort", "answer", "response", "completion",
  "output", "assistant", "ki"
];

const STOP_WORDS = new Set([
  "der", "die", "das", "den", "dem", "des",
  "ein", "eine", "einer", "eines", "einem",
  "und", "oder", "aber", "auch", "ist", "sind",
  "war", "waren", "bin", "bist", "du", "ich",
  "er", "sie", "es", "wir", "ihr", "man",
  "mit", "von", "für", "auf", "im", "in", "am",
  "an", "zu", "zum", "zur", "bei", "nach",
  "wie", "was", "wer", "wo", "wann", "warum",
  "wieso", "welche", "welcher", "welches"
]);

function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}

function finite(x) {
  return typeof x === "number" && Number.isFinite(x);
}

function vector(n, value = 0) {
  return Array.from({ length: n }, () => value);
}

function matrix(rows, cols, value = 0) {
  return Array.from(
    { length: rows },
    () => vector(cols, value)
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

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }

  return array;
}

function softmax(logits, temperature = 1) {
  if (!logits.length) return [];

  temperature = clamp(temperature, 0.05, 5);

  let max = -Infinity;

  for (const value of logits) {
    if (finite(value) && value > max) max = value;
  }

  if (!finite(max)) {
    return vector(logits.length, 1 / logits.length);
  }

  const result = new Array(logits.length);
  let total = 0;

  for (let i = 0; i < logits.length; i++) {
    const value = finite(logits[i])
      ? Math.exp(clamp((logits[i] - max) / temperature, -60, 0))
      : 0;

    result[i] = value;
    total += value;
  }

  if (!finite(total) || total <= 0) {
    return vector(logits.length, 1 / logits.length);
  }

  for (let i = 0; i < result.length; i++) {
    result[i] /= total;
  }

  return result;
}

function sample(probabilities) {
  let total = 0;

  for (const p of probabilities) {
    if (finite(p) && p > 0) total += p;
  }

  if (total <= 0) return -1;

  let r = Math.random() * total;

  for (let i = 0; i < probabilities.length; i++) {
    const p = probabilities[i];

    if (!finite(p) || p <= 0) continue;

    r -= p;

    if (r <= 0) return i;
  }

  return probabilities.length - 1;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

class NeuronalesNetz {
  constructor(tokenizer = null, optionen = {}) {
    if (tokenizer && typeof tokenizer.zerlege !== "function") {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.tokenizer = tokenizer;

    this.optionen = {
      maxVokabular: 256,
      embeddingGroesse: 32,
      versteckteNeuronen: 250,
      kontextLaenge: 32,
      segmentLaenge: 16,
      maxTrainingsTokens: 12000,
      ngramOrdnung: 4,
      lernrate: 0.015,
      epochen: 4,
      gradientGrenze: 1,
      maxWortZeichen: 48
    };

    for (const [key, value] of Object.entries(optionen || {})) {
      if (key in this.optionen && finite(value)) {
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
      6
    );

    this.optionen.lernrate = clamp(
      this.optionen.lernrate,
      0.00001,
      0.1
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

    this.ngramCounts = new Map();
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
    this.konversationsModus = true;

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
    return this.normalisiereText(text).match(TOKEN_RE) || [];
  }

  charToken(char) {
    return `<CHAR:${char.codePointAt(0).toString(16)}>`;
  }

  charAusToken(token) {
    const match = CHAR_RE.exec(token || "");

    if (!match) return null;

    try {
      return String.fromCodePoint(parseInt(match[1], 16));
    } catch {
      return null;
    }
  }

  istWortToken(token) {
    return WORD_RE.test(token);
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
    const gesehen = new Set();

    const besuchen = wert => {
      if (typeof wert === "string") {
        const text = this.normalisiereText(wert);

        if (text) {
          const key = "text:" + text;

          if (!gesehen.has(key)) {
            gesehen.add(key);
            sequenzen.push({ art: "text", text });
          }
        }

        return;
      }

      if (Array.isArray(wert)) {
        for (const element of wert) besuchen(element);
        return;
      }

      if (!wert || typeof wert !== "object") return;

      const frageFeld = this.findeFeld(wert, QUESTION_FIELDS);
      const antwortFeld = this.findeFeld(wert, ANSWER_FIELDS);

      if (
        frageFeld &&
        antwortFeld &&
        typeof wert[frageFeld] === "string" &&
        typeof wert[antwortFeld] === "string"
      ) {
        const frage = this.normalisiereText(wert[frageFeld]);
        const antwort = this.normalisiereText(wert[antwortFeld]);

        if (frage && antwort) {
          const key = "dialog:" + frage + "\n" + antwort;

          if (!gesehen.has(key)) {
            gesehen.add(key);

            const paar = { frage, antwort };
            paare.push(paar);
            sequenzen.push({ art: "dialog", frage, antwort });
          }
        }

        return;
      }

      for (const unterwert of Object.values(wert)) {
        besuchen(unterwert);
      }
    };

    besuchen(daten);

    return { sequenzen, paare };
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
    const sonderTokens = new Set();
    const zeichen = new Set();

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

    const charTokens = [...zeichen].map(char => this.charToken(char));

    this.vokabular = [...new Set([
      ...SPECIAL_LIST,
      ...woerter,
      ...sonderTokens,
      ...charTokens
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
        const id = this.tokenZuId.get(this.charToken(char));
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
      V, E, Math.sqrt(1 / E)
    );

    // Eingabegewichte: H Zeilen, E Spalten.
    this.gewichteEingabe = randomMatrix(
      H, E, Math.sqrt(2 / (E + H))
    );

    // Rekurrenz: H x H.
    this.gewichteRekurrenz = randomMatrix(
      H, H, Math.sqrt(1 / H)
    );

    this.biasVersteckt = vector(H);

    // Ausgabe: V Zeilen, H Spalten.
    this.gewichteAusgabe = randomMatrix(
      V, H, Math.sqrt(2 / (H + V))
    );

    this.biasAusgabe = vector(V);
  }

  berechneSchritt(tokenId, vorherigerZustand) {
    const H = this.versteckteNeuronen;
    const E = this.embeddingGroesse;
    const V = this.vokabular.length;

    const sicherId = clamp(
      Math.floor(tokenId),
      0,
      Math.max(0, V - 1)
    );

    const embedding = this.embeddings[sicherId];
    const zustand = vector(H);

    for (let h = 0; h < H; h++) {
      let summe = this.biasVersteckt[h];

      for (let e = 0; e < E; e++) {
        summe += this.gewichteEingabe[h][e] * embedding[e];
      }

      for (let j = 0; j < H; j++) {
        summe +=
          this.gewichteRekurrenz[h][j] *
          vorherigerZustand[j];
      }

      zustand[h] = Math.tanh(clamp(summe, -20, 20));
    }

    const logits = vector(V);

    for (let v = 0; v < V; v++) {
      let summe = this.biasAusgabe[v];

      for (let h = 0; h < H; h++) {
        summe += this.gewichteAusgabe[v][h] * zustand[h];
      }

      logits[v] = clamp(summe, -30, 30);
    }

    return {
      tokenId: sicherId,
      embedding,
      vorherigerZustand,
      zustand,
      logits,
      wahrscheinlichkeiten: softmax(logits)
    };
  }

  berechneVorwaerts(tokenIds, trainingsModus = false) {
    let zustand = vector(this.versteckteNeuronen);
    const schritte = [];

    for (const tokenId of tokenIds) {
      const schritt = this.berechneSchritt(tokenId, zustand);
      zustand = schritt.zustand;

      if (trainingsModus) schritte.push(schritt);
    }

    const letzter = this.vorhersageLogits(zustand);

    return {
      zustand,
      schritte,
      logits: letzter,
      wahrscheinlichkeiten: softmax(letzter)
    };
  }

  vorhersageLogits(zustand) {
    const V = this.vokabular.length;
    const H = this.versteckteNeuronen;
    const logits = vector(V);

    for (let v = 0; v < V; v++) {
      let summe = this.biasAusgabe[v];

      for (let h = 0; h < H; h++) {
        summe += this.gewichteAusgabe[v][h] * zustand[h];
      }

      logits[v] = clamp(summe, -30, 30);
    }

    return logits;
  }

  berechneVerlust(probabilities, zielId) {
    return -Math.log(
      clamp(probabilities[zielId] ?? 1e-9, 1e-9, 1)
    );
  }

  berechneGradienten(schritte, zielIds) {
    const H = this.versteckteNeuronen;
    const E = this.embeddingGroesse;
    const V = this.vokabular.length;

    const grad = {
      embeddings: matrix(V, E),
      gewichteEingabe: matrix(H, E),
      gewichteRekurrenz: matrix(H, H),
      biasVersteckt: vector(H),
      gewichteAusgabe: matrix(V, H),
      biasAusgabe: vector(V),
      verlust: 0,
      anzahl: 0
    };

    let deltaZustandNaechster = vector(H);

    for (let t = schritte.length - 1; t >= 0; t--) {
      const schritt = schritte[t];
      const zielId = zielIds[t];

      if (zielId == null || zielId < 0 || zielId >= V) {
        continue;
      }

      const deltaAusgabe = schritt.wahrscheinlichkeiten.slice();
      deltaAusgabe[zielId] -= 1;

      grad.verlust += this.berechneVerlust(
        schritt.wahrscheinlichkeiten,
        zielId
      );

      grad.anzahl++;

      for (let v = 0; v < V; v++) {
        const delta = deltaAusgabe[v];

        grad.biasAusgabe[v] += delta;

        for (let h = 0; h < H; h++) {
          grad.gewichteAusgabe[v][h] +=
            delta * schritt.zustand[h];
        }
      }

      const deltaHidden = vector(H);

      for (let h = 0; h < H; h++) {
        let summe = deltaZustandNaechster[h];

        for (let v = 0; v < V; v++) {
          summe +=
            this.gewichteAusgabe[v][h] *
            deltaAusgabe[v];
        }

        deltaHidden[h] =
          summe * (1 - schritt.zustand[h] ** 2);
      }

      const deltaVorherigerZustand = vector(H);

      for (let h = 0; h < H; h++) {
        const delta = deltaHidden[h];

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

          deltaVorherigerZustand[j] +=
            this.gewichteRekurrenz[h][j] * delta;
        }
      }

      deltaZustandNaechster = deltaVorherigerZustand;
    }

    return grad;
  }

  wendeGradientenAn(grad) {
    const faktor =
      this.optionen.lernrate / Math.max(1, grad.anzahl);

    const grenze = this.optionen.gradientGrenze;

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
          const delta = clamp(
            quelle[i][j] * faktor,
            -grenze,
            grenze
          );

          ziel[i][j] = clamp(
            ziel[i][j] - delta,
            -8,
            8
          );
        }
      }
    }

    for (const name of ["biasVersteckt", "biasAusgabe"]) {
      const ziel = this[name];
      const quelle = grad[name];

      for (let i = 0; i < ziel.length; i++) {
        ziel[i] = clamp(
          ziel[i] - clamp(quelle[i] * faktor, -grenze, grenze),
          -8,
          8
        );
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

    let verlustSumme = 0;
    let tokenSumme = 0;

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

      const schritte = [];
      let zustand = vector(this.versteckteNeuronen);

      for (const id of eingabe) {
        const schritt = this.berechneSchritt(id, zustand);
        schritte.push(schritt);
        zustand = schritt.zustand;
      }

      const grad = this.berechneGradienten(schritte, ziele);

      this.wendeGradientenAn(grad);

      verlustSumme += grad.verlust;
      tokenSumme += grad.anzahl;
    }

    return {
      verlust: tokenSumme ? verlustSumme / tokenSumme : 0,
      tokens: tokenSumme
    };
  }

  aktualisiereNGramme(tokens) {
    for (let i = 0; i < tokens.length; i++) {
      const ziel = tokens[i];

      for (let n = 0; n <= this.ngramOrdnung; n++) {
        const start = Math.max(0, i - n);
        const kontext = tokens.slice(start, i).join(" ");

        if (!this.ngramCounts.has(kontext)) {
          this.ngramCounts.set(kontext, new Map());
        }

        const ziele = this.ngramCounts.get(kontext);
        ziele.set(ziel, (ziele.get(ziel) || 0) + 1);
      }
    }
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
    const daten = this.trainingsSequenzen.slice();
    shuffle(daten);

    let verlustSumme = 0;
    let tokenSumme = 0;

    for (const sequenz of daten) {
      const ergebnis = this.trainiereEinBeispiel(sequenz);

      verlustSumme += ergebnis.verlust * ergebnis.tokens;
      tokenSumme += ergebnis.tokens;
    }

    return {
      verlust: tokenSumme ? verlustSumme / tokenSumme : 0,
      tokens: tokenSumme
    };
  }

  trainiere(daten, optionen = {}) {
    try {
      this.letzterFehler = null;

      if (!Array.isArray(daten) || daten.length === 0) {
        throw new Error("Keine Trainingsdaten vorhanden.");
      }

      if (finite(optionen.lernrate)) {
        this.optionen.lernrate = clamp(
          optionen.lernrate,
          0.00001,
          0.1
        );
      }

      const gesammelt = this.sammleTrainingsDaten(daten);

      if (!gesammelt.sequenzen.length) {
        throw new Error("Keine verwendbaren Texte gefunden.");
      }

      let sequenzen = gesammelt.sequenzen;

      if (finite(optionen.maxTrainingsBeispiele)) {
        const limit = Math.max(
          1,
          Math.floor(optionen.maxTrainingsBeispiele)
        );

        sequenzen = sequenzen.slice(0, limit);
      }

      const paare = gesammelt.paare;

      this.baueVokabular(sequenzen);
      this.initialisiereGewichte();

      this.trainingsPaare = paare;
      this.trainingsSequenzen = sequenzen
        .map(s => this.kodiereSequenz(s))
        .filter(ids => ids.length >= 2);

      if (!this.trainingsSequenzen.length) {
        throw new Error("Die Trainingssequenzen sind leer.");
      }

      this.trainingsBeispiele = this.trainingsSequenzen.length;
      this.trainierteEpochen = 0;
      this.verlustHistorie = [];

      this.trainiereNGramModelle();

      const epochen = clamp(
        Math.floor(optionen.epochen ?? this.optionen.epochen),
        1,
        100
      );

      for (let e = 0; e < epochen; e++) {
        const ergebnis = this.trainiereEpoche();

        this.trainierteEpochen++;

        this.verlustHistorie.push({
          epoche: this.trainierteEpochen,
          verlust: ergebnis.verlust,
          tokens: ergebnis.tokens
        });

        console.log(
          `RNN-Training: Epoche ${this.trainierteEpochen}/${epochen}, ` +
          `${ergebnis.tokens} Token-Schritte, ` +
          `Verlust ${ergebnis.verlust.toFixed(3)}`
        );
      }

      this.bereit = true;

      return {
        erfolg: true,
        beispiele: this.trainingsBeispiele,
        dialogPaare: this.trainingsPaare.length,
        vokabularGroesse: this.vokabular.length,
        versteckteNeuronen: this.versteckteNeuronen,
        epochen: this.trainierteEpochen,
        verlust: this.verlustHistorie.at(-1)?.verlust ?? null
      };
    } catch (fehler) {
      this.letzterFehler = fehler.message;
      this.bereit = false;

      return {
        erfolg: false,
        fehler: fehler.message
      };
    }
  }

  // Kompatibel mit dem vorhandenen server.js.
  trainiereTexte(daten, tokenizer = this.tokenizer, optionen = {}) {
    if (
      tokenizer &&
      typeof tokenizer === "object" &&
      typeof tokenizer.zerlege !== "function" &&
      arguments.length === 2
    ) {
      optionen = tokenizer;
      tokenizer = this.tokenizer;
    }

    if (tokenizer) this.tokenizer = tokenizer;

    return this.trainiere(daten, optionen);
  }

  waehleToken(probabilities, bisherigeIds = [], optionen = {}) {
    const temperatur = clamp(optionen.temperatur ?? 0.7, 0.1, 2);
    const topK = clamp(
      Math.floor(optionen.topK ?? 8),
      1,
      100
    );

    const verboten = new Set([
      this.padId,
      this.bosId,
      this.userId,
      this.aiId,
      this.wordId,
      this.endWordId
    ]);

    const kandidaten = [];

    for (let id = 0; id < probabilities.length; id++) {
      if (verboten.has(id)) continue;

      const token = this.vokabular[id];

      if (!token) continue;

      let p = probabilities[id];

      if (!finite(p) || p <= 0) continue;

      const letzte = bisherigeIds.slice(-4);

      if (letzte.length >= 3 && letzte.every(x => x === id)) {
        p *= 0.05;
      }

      if (id === this.eosId) p *= 1.4;

      p = Math.pow(p, 1 / temperatur);

      kandidaten.push({ id, p });
    }

    kandidaten.sort((a, b) => b.p - a.p);

    const top = kandidaten.slice(0, topK);

    if (!top.length) return this.unkId;

    const summe = top.reduce((s, x) => s + x.p, 0);

    if (summe <= 0) return top[0].id;

    const gewichte = top.map(x => x.p / summe);
    const index = sample(gewichte);

    return index < 0 ? top[0].id : top[index].id;
  }

  dekodiereIds(ids) {
    const tokens = [];
    let zeichenWort = null;

    for (const id of ids) {
      const token = this.vokabular[id];

      if (!token) continue;

      if (token === SPECIAL.EOS) break;

      if (token === SPECIAL.WORD) {
        zeichenWort = "";
        continue;
      }

      if (token === SPECIAL.END_WORD) {
        if (zeichenWort !== null) tokens.push(zeichenWort);
        zeichenWort = null;
        continue;
      }

      const char = this.charAusToken(token);

      if (char !== null) {
        if (zeichenWort !== null) {
          zeichenWort += char;
        } else {
          tokens.push(char);
        }

        continue;
      }

      if (SPECIAL_LIST.includes(token)) continue;

      if (zeichenWort !== null) {
        tokens.push(zeichenWort);
        zeichenWort = null;
      }

      tokens.push(token);
    }

    if (zeichenWort !== null) tokens.push(zeichenWort);

    let text = "";

    for (const token of tokens) {
      if (!text) {
        text = token;
      } else if (/^[.,!?;:%)\]}…»]/u.test(token)) {
        text += token;
      } else if (/^['’]/u.test(token)) {
        text += token;
      } else {
        text += " " + token;
      }
    }

    return text
      .replace(/\s+([.,!?;:%)\]}»])/gu, "$1")
      .replace(/([([{«])\s+/gu, "$1")
      .replace(/\s+/gu, " ")
      .trim();
  }

  generiereMitNetz(prompt, optionen = {}) {
    const promptTokens = [
      SPECIAL.BOS,
      SPECIAL.USER,
      ...this.zerlegeText(prompt),
      SPECIAL.AI
    ];

    let kontext = [];

    for (const token of promptTokens) {
      kontext.push(...this.kodiereEinToken(token));
    }

    kontext = kontext.slice(-this.kontextLaenge);

    const maxTokens = clamp(
      Math.floor(optionen.maxTokens ?? 40),
      1,
      100
    );

    const generierteIds = [];

    for (let i = 0; i < maxTokens; i++) {
      const vorwaerts = this.berechneVorwaerts(kontext);

      const naechsteId = this.waehleToken(
        softmax(
          vorwaerts.logits,
          optionen.temperatur ?? 0.7
        ),
        generierteIds,
        optionen
      );

      if (
        naechsteId === this.eosId ||
        naechsteId === this.padId ||
        naechsteId === this.bosId
      ) {
        break;
      }

      generierteIds.push(naechsteId);

      kontext.push(naechsteId);
      kontext = kontext.slice(-this.kontextLaenge);
    }

    return this.dekodiereIds(generierteIds);
  }

  wichtigeWoerter(text) {
    return new Set(
      this.zerlegeText(text)
        .map(x => x.toLocaleLowerCase("de-DE"))
        .filter(x =>
          /[\p{L}\p{N}]/u.test(x) &&
          x.length > 2 &&
          !STOP_WORDS.has(x) &&
          !x.startsWith("<")
        )
    );
  }

  aehnlichkeit(textA, textB) {
    const a = this.wichtigeWoerter(textA);
    const b = this.wichtigeWoerter(textB);

    if (!a.size || !b.size) {
      return this.normalisiereText(textA).toLowerCase() ===
        this.normalisiereText(textB).toLowerCase()
        ? 1
        : 0;
    }

    let gemeinsam = 0;

    for (const wort of a) {
      if (b.has(wort)) gemeinsam++;
    }

    const vereinigt = new Set([...a, ...b]).size;
    const jaccard = gemeinsam / Math.max(1, vereinigt);
    const abdeckung = gemeinsam / Math.max(1, a.size);

    return 0.6 * jaccard + 0.4 * abdeckung;
  }

  findePassendesBeispiel(frage) {
    let bestesPaar = null;
    let bestePunktzahl = 0;

    const normalisiert = this.normalisiereText(frage).toLowerCase();

    for (const paar of this.trainingsPaare) {
      const fragePaar = this.normalisiereText(paar.frage).toLowerCase();

      const punktzahl = normalisiert === fragePaar
        ? 1
        : this.aehnlichkeit(normalisiert, fragePaar);

      if (punktzahl > bestePunktzahl) {
        bestePunktzahl = punktzahl;
        bestesPaar = paar;
      }
    }

    return {
      paar: bestesPaar,
      punktzahl: bestePunktzahl
    };
  }

  antworte(frage, optionen = {}) {
    try {
      if (!this.bereit) {
        return {
          text: "Ich bin noch nicht trainiert. Bitte trainiere mich zuerst.",
          methode: "status"
        };
      }

      const prompt = this.normalisiereText(frage);

      if (!prompt) {
        return {
          text: "Bitte gib eine Nachricht ein.",
          methode: "validierung"
        };
      }

      const treffer = this.findePassendesBeispiel(prompt);

      // Bei einer wirklich ähnlichen Trainingsfrage ist die
      // gespeicherte Antwort meist zuverlässiger als freie Generierung.
      if (
        treffer.paar &&
        treffer.punktzahl >= (optionen.mindestAehnlichkeit ?? 0.72) &&
        !optionen.freieGenerierung
      ) {
        const ergebnis = {
          text: treffer.paar.antwort,
          methode: "trainingsbeispiel",
          aehnlichkeit: Number(treffer.punktzahl.toFixed(4))
        };

        this.letzteAntwortAnalyse = ergebnis;
        return ergebnis;
      }

      if (optionen.freieGenerierung || !treffer.paar) {
        const text = this.generiereMitNetz(prompt, optionen);

        const ergebnis = {
          text: text ||
            "Ich habe noch keine gute Antwort gelernt. Trainiere mich mit weiteren Beispielen.",
          methode: "rnn",
          aehnlichkeit: Number(treffer.punktzahl.toFixed(4))
        };

        this.letzteAntwortAnalyse = ergebnis;
        return ergebnis;
      }

      const ergebnis = {
        text: treffer.paar.antwort,
        methode: "trainingsbeispiel",
        aehnlichkeit: Number(treffer.punktzahl.toFixed(4))
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

  // Kompatibilität mit deinem bisherigen server.js.
  antwortGenerieren(prompt, optionen = {}) {
    const ergebnis = this.antworte(prompt, {
      ...optionen,
      freieGenerierung: true
    });

    return ergebnis.text;
  }

  statistik() {
    return {
      version: VERSION,
      bereit: this.bereit,
      vokabularGroesse: this.vokabular.length,
      trainingsBeispiele: this.trainingsBeispiele,
      dialogPaare: this.trainingsPaare.length,
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

  status() {
    return this.statistik();
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
      trainingsSequenzen: this.trainingsSequenzen,
      trainierteEpochen: this.trainierteEpochen,
      verlustHistorie: this.verlustHistorie,
      bereit: this.bereit
    };

    fs.mkdirSync(path.dirname(ziel), { recursive: true });
    fs.writeFileSync(temporaer, JSON.stringify(daten), "utf8");
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
          "Die Modelldatei hat eine andere Version. Bitte neu trainieren."
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
      this.wordId = this.tokenZuId.get(SPECIAL.WORD) ?? 6;
      this.endWordId = this.tokenZuId.get(SPECIAL.END_WORD) ?? 7;

      this.embeddings = daten.embeddings;
      this.gewichteEingabe = daten.gewichteEingabe;
      this.gewichteRekurrenz = daten.gewichteRekurrenz;
      this.biasVersteckt = daten.biasVersteckt;
      this.gewichteAusgabe = daten.gewichteAusgabe;
      this.biasAusgabe = daten.biasAusgabe;

      this.trainingsPaare = daten.trainingsPaare || [];
      this.trainingsSequenzen = daten.trainingsSequenzen || [];
      this.trainingsBeispiele = this.trainingsSequenzen.length;
      this.trainierteEpochen = daten.trainierteEpochen || 0;
      this.verlustHistorie = daten.verlustHistorie || [];

      this.ngramCounts = new Map();
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
