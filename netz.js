"use strict";

/*
 * Lokales generatives Sprachmodell für Node.js.
 * - Keine API und keine externen Pakete erforderlich.
 * - Verwendet bestehende JSON-Trainingsdaten (Textsammlungen und Frage/Antwort-Paare).
 * - Trainiert ein kleines rekurrentes neuronales Netz (RNN) auf nächstes-Token-Vorhersage.
 * - Antwortgenerierung erfolgt Token für Token, nicht durch Nachschlagen einer fertigen Antwort.
 *
 * Hinweis: Ein kleines RNN ist kein ChatGPT-Ersatz. Es benötigt Trainingsdaten und kann
 * trotz korrektem Training noch Fehler und unlogische Antworten erzeugen.
 */

const fs = require("node:fs");
const path = require("node:path");

const SPECIAL = {
  PAD: "<PAD>",
  BOS: "<BOS>",
  EOS: "<EOS>",
  UNK: "<UNK>",
  USER: "<benutzer>",
  AI: "<ki>",
  WORD: "<WORTZEICHEN>",
  END_WORD: "</WORTZEICHEN>"
};

const SPECIAL_TOKENS = Object.values(SPECIAL);
const WORD_RE = /^[\p{L}\p{M}\p{N}_]+(?:['’.-][\p{L}\p{M}\p{N}_]+)*$/u;
const TOKEN_RE = /<[^>\s]+>|[\p{L}\p{M}\p{N}_]+(?:['’.-][\p{L}\p{M}\p{N}_]+)*|[^\s]/gu;
const FIELD_QUESTION = ["frage", "question", "prompt", "input", "user", "benutzer"];
const FIELD_ANSWER = ["antwort", "answer", "response", "completion", "output", "assistant", "ki"];

function zeros(length) {
  return Array(length).fill(0);
}

function matrix(rows, cols, fill = 0) {
  return Array.from({ length: rows }, () => Array(cols).fill(fill));
}

function randomMatrix(rows, cols, scale) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
  );
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function cloneArray(array) {
  return Array.isArray(array) ? array.map(value =>
    Array.isArray(value) ? value.slice() : value
  ) : array;
}

class NeuronalesNetz {
  constructor(tokenizer = null, optionen = {}) {
    // Rückwärtskompatibel: new NeuronalesNetz(optionen) ist ebenfalls erlaubt.
    if (tokenizer && typeof tokenizer.zerlege !== "function") {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.tokenizer = tokenizer || null;
    this.maxVokabular = Math.max(80, Math.floor(optionen.maxVokabular || 420));
    this.embeddingGroesse = Math.max(8, Math.floor(optionen.embeddingGroesse || 20));
    this.versteckteNeuronen = Math.max(16, Math.floor(optionen.versteckteNeuronen || 32));
    this.kontextLaenge = Math.max(4, Math.floor(optionen.kontextLaenge || 32));
    this.trainingsSchrittLaenge = Math.max(4, Math.floor(optionen.trainingsSchrittLaenge || 12));
    this.maxTrainingsTokens = Math.max(100, Math.floor(optionen.maxTrainingsTokens || 7000));

    this.vokabular = [];
    this.tokenZuId = new Map();
    this.haeufigeWoerter = new Set();
    this.zeichenTokens = new Set();
    this.embeddings = null;
    this.gewichteEingabe = null;
    this.gewichteRekurrenz = null;
    this.biasVersteckt = null;
    this.gewichteAusgabe = null;
    this.biasAusgabe = null;

    this.padId = 0;
    this.bosId = 1;
    this.eosId = 2;
    this.unkId = 3;
    this.userId = 4;
    this.aiId = 5;
    this.wordId = 6;
    this.endWordId = 7;

    this.bereit = false;
    this.konversationsModus = false;
    this.trainingsBeispiele = 0;
    this.trainierteEpochen = 0;
    this.trainingsPaare = [];
    this.trainingsSequenzen = [];
    this.letzteAntworten = [];
    this.letzterFehler = null;
    this.letzteAntwortAnalyse = null;
    this.verlustHistorie = [];
    this.wortHaeufigkeit = new Map();
    this.ngramOrdnung = Math.max(2, Math.min(7, Math.floor(optionen.ngramOrdnung || 5)));
    this.ngramCounts = Array.from({ length: this.ngramOrdnung + 1 }, () => new Map());
  }

  normalisiereText(text) {
    return String(text ?? "").normalize("NFC").trim();
  }

  zerlegeText(text) {
    const sauber = this.normalisiereText(text);
    return sauber.match(TOKEN_RE) || [];
  }

  findeFeld(objekt, namen) {
    const lowerToOriginal = new Map(
      Object.keys(objekt).map(key => [key.toLocaleLowerCase("de-DE"), key])
    );
    for (const name of namen) {
      const gefunden = lowerToOriginal.get(name);
      if (gefunden !== undefined) return gefunden;
    }
    return null;
  }

  sammleTrainingsDaten(daten) {
    const sequenzen = [];
    const paare = [];

    const untersuche = wert => {
      if (typeof wert === "string") {
        const text = this.normalisiereText(wert);
        if (text) sequenzen.push({ art: "text", text });
        return;
      }
      if (Array.isArray(wert)) {
        for (const eintrag of wert) untersuche(eintrag);
        return;
      }
      if (!wert || typeof wert !== "object") return;

      const frageFeld = this.findeFeld(wert, FIELD_QUESTION);
      const antwortFeld = this.findeFeld(wert, FIELD_ANSWER);
      if (
        frageFeld && antwortFeld &&
        typeof wert[frageFeld] === "string" &&
        typeof wert[antwortFeld] === "string"
      ) {
        const frage = this.normalisiereText(wert[frageFeld]);
        const antwort = this.normalisiereText(wert[antwortFeld]);
        if (frage && antwort) {
          paare.push({ frage, antwort });
          sequenzen.push({ art: "dialog", frage, antwort });
        }
        return;
      }

      for (const unterwert of Object.values(wert)) untersuche(unterwert);
    };

    untersuche(daten);
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
    return [SPECIAL.BOS, ...this.zerlegeText(sequenz.text), SPECIAL.EOS];
  }

  baueVokabular(sequenzen) {
    const haeufigkeit = new Map();
    const zeichen = new Set();
    const sonderzeichen = new Set();

    for (const sequenz of sequenzen) {
      const tokens = this.tokensDerSequenz(sequenz);
      for (const token of tokens) {
        if (SPECIAL_TOKENS.includes(token)) continue;
        haeufigkeit.set(token, (haeufigkeit.get(token) || 0) + 1);
        if (WORD_RE.test(token)) {
          for (const zeichenEinzeln of Array.from(token)) zeichen.add(zeichenEinzeln);
        } else {
          sonderzeichen.add(token);
        }
      }
    }

    const woerterSortiert = [...haeufigkeit.entries()]
      .filter(([token]) => WORD_RE.test(token))
      .sort((a, b) => b[1] - a[1]);

    const haeufige = woerterSortiert
      .slice(0, this.maxVokabular)
      .map(([token]) => token);

    this.haeufigeWoerter = new Set(haeufige);
    this.zeichenTokens = zeichen;
    this.wortHaeufigkeit = haeufigkeit;

    const chars = [...zeichen].map(char => this.charToken(char));
    const alle = [
      ...SPECIAL_TOKENS,
      ...haeufige,
      ...[...sonderzeichen].filter(token => !SPECIAL_TOKENS.includes(token)),
      ...chars
    ];

    this.vokabular = [...new Set(alle)];
    this.tokenZuId = new Map(this.vokabular.map((token, id) => [token, id]));

    this.padId = this.tokenZuId.get(SPECIAL.PAD);
    this.bosId = this.tokenZuId.get(SPECIAL.BOS);
    this.eosId = this.tokenZuId.get(SPECIAL.EOS);
    this.unkId = this.tokenZuId.get(SPECIAL.UNK);
    this.userId = this.tokenZuId.get(SPECIAL.USER);
    this.aiId = this.tokenZuId.get(SPECIAL.AI);
    this.wordId = this.tokenZuId.get(SPECIAL.WORD);
    this.endWordId = this.tokenZuId.get(SPECIAL.END_WORD);
  }

  charToken(zeichen) {
    return `<ZEICHEN:${zeichen.codePointAt(0).toString(16)}>`;
  }

  charAusToken(token) {
    const treffer = /^<ZEICHEN:([0-9a-f]+)>$/i.exec(token || "");
    if (!treffer) return null;
    try {
      return String.fromCodePoint(parseInt(treffer[1], 16));
    } catch {
      return null;
    }
  }

  editierDistanz(a, b, limit = 3) {
    a = String(a);
    b = String(b);
    if (Math.abs(a.length - b.length) > limit) return limit + 1;
    let vorher = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      const aktuell = [i];
      let minimum = i;
      for (let j = 1; j <= b.length; j++) {
        const kosten = a[i - 1] === b[j - 1] ? 0 : 1;
        aktuell[j] = Math.min(aktuell[j - 1] + 1, vorher[j] + 1, vorher[j - 1] + kosten);
        minimum = Math.min(minimum, aktuell[j]);
      }
      if (minimum > limit) return limit + 1;
      vorher = aktuell;
    }
    return vorher[b.length];
  }

  findeBekanntesWort(token, korrigiereTippfehler = false) {
    const direkt = this.tokenZuId.get(token);
    if (direkt !== undefined) return direkt;
    if (!WORD_RE.test(token)) return undefined;

    const lower = token.toLocaleLowerCase("de-DE");
    const varianten = [];
    for (const bekannt of this.haeufigeWoerter) {
      if (bekannt.toLocaleLowerCase("de-DE") === lower) varianten.push(bekannt);
    }
    if (varianten.length) return this.tokenZuId.get(varianten[0]);

    if (!korrigiereTippfehler || lower.length < 5) return undefined;
    const limit = Math.max(1, Math.floor(lower.length * 0.15));
    let bestesWort = null;
    let besteDistanz = limit + 1;
    let gleichstand = false;
    for (const bekannt of this.haeufigeWoerter) {
      const knownLower = bekannt.toLocaleLowerCase("de-DE");
      if (knownLower.length < 4 || Math.abs(knownLower.length - lower.length) > limit) continue;
      const distanz = this.editierDistanz(lower, knownLower, limit);
      if (distanz < besteDistanz) {
        besteDistanz = distanz;
        bestesWort = bekannt;
        gleichstand = false;
      } else if (distanz === besteDistanz && bekannt !== bestesWort) {
        gleichstand = true;
      }
    }
    if (!bestesWort || besteDistanz > limit || gleichstand) return undefined;
    return this.tokenZuId.get(bestesWort);
  }

  kodierePrompt(text) {
    const ids = [];
    for (const token of this.zerlegeText(text)) {
      const known = this.findeBekanntesWort(token, true);
      if (known !== undefined) ids.push(known);
      else ids.push(...this.kodiereEinToken(token));
    }
    return ids;
  }

  kodiereEinToken(token) {
    const direct = this.tokenZuId.get(token);
    if (direct !== undefined) return [direct];

    if (WORD_RE.test(token)) {
      const result = [this.wordId];
      for (const char of Array.from(token)) {
        const charId = this.tokenZuId.get(this.charToken(char));
        result.push(charId === undefined ? this.unkId : charId);
      }
      result.push(this.endWordId);
      return result;
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
    const embScale = Math.sqrt(1 / E);
    const wxhScale = Math.sqrt(2 / (E + H));
    const whhScale = Math.sqrt(1 / H);
    const whyScale = Math.sqrt(2 / (H + V));

    this.embeddings = randomMatrix(V, E, embScale);
    this.gewichteEingabe = randomMatrix(E, H, wxhScale);
    this.gewichteRekurrenz = randomMatrix(H, H, whhScale);
    this.biasVersteckt = zeros(H);
        this.gewichteAusgabe = randomMatrix(H, V, whyScale);
    this.biasAusgabe = zeros(V);
  }

  baueNgramModell(kodierteSequenzen) {
    this.ngramCounts = Array.from({ length: this.ngramOrdnung + 1 }, () => new Map());
    for (const ids of kodierteSequenzen) {
      for (let i = 1; i < ids.length; i++) {
        const next = ids[i];
        const maxOrder = Math.min(this.ngramOrdnung, i);
        for (let order = 1; order <= maxOrder; order++) {
          const key = ids.slice(i - order, i).join(",");
          let distribution = this.ngramCounts[order].get(key);
          if (!distribution) {
            distribution = new Map();
            this.ngramCounts[order].set(key, distribution);
          }
          distribution.set(next, (distribution.get(next) || 0) + 1);
        }
      }
    }
  }

  findeNgramVerteilung(kontextIds, erlaubt) {
    if (!this.ngramCounts || !this.ngramCounts.length) return null;
    const maxOrder = Math.min(this.ngramOrdnung, kontextIds.length);
    for (let order = maxOrder; order >= 1; order--) {
      const key = kontextIds.slice(-order).join(",");
      const distribution = this.ngramCounts[order]?.get(key);
      if (!distribution || distribution.size === 0) continue;
      let sum = 0;
      for (const [id, count] of distribution) {
        if (erlaubt(id)) sum += count;
      }
      if (!sum) continue;
      const probabilities = new Map();
      for (const [id, count] of distribution) {
        if (erlaubt(id)) probabilities.set(id, count / sum);
      }
      const alpha = Math.min(0.66, 0.12 + 0.115 * order);
      return { probabilities, alpha, order, occurrences: sum };
    }
    return null;
  }

  softmax(logits, temperatur = 1) {
    const temp = Math.max(0.1, Math.min(2.5, temperatur));
    let max = -Infinity;
    for (const value of logits) if (value > max) max = value;
    const result = new Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      const value = Math.exp(Math.max(-60, (logits[i] - max) / temp));
      result[i] = value;
      sum += value;
    }
    if (!Number.isFinite(sum) || sum <= 0) return zeros(logits.length).map(() => 1 / logits.length);
    for (let i = 0; i < result.length; i++) result[i] /= sum;
    return result;
  }

  schrittVorwaerts(tokenId, hVorher, temperatur = 1) {
    const E = this.embeddingGroesse;
    const H = this.versteckteNeuronen;
    const V = this.vokabular.length;
    const embedding = this.embeddings[tokenId] || this.embeddings[this.unkId];
    const h = new Array(H);

    for (let j = 0; j < H; j++) {
      let wert = this.biasVersteckt[j];
      for (let i = 0; i < E; i++) wert += embedding[i] * this.gewichteEingabe[i][j];
      for (let i = 0; i < H; i++) wert += hVorher[i] * this.gewichteRekurrenz[i][j];
      h[j] = Math.tanh(wert);
    }

    const logits = new Array(V);
    for (let v = 0; v < V; v++) {
      let wert = this.biasAusgabe[v];
      for (let j = 0; j < H; j++) wert += h[j] * this.gewichteAusgabe[j][v];
      logits[v] = wert;
    }

    return {
      h,
      hVorher: hVorher.slice(),
      tokenId,
      wahrscheinlichkeiten: this.softmax(logits, temperatur)
    };
  }

  begrenzeGradient(wert) {
    return Math.max(-3, Math.min(3, wert));
  }

  trainiereAbschnitt(ids, start, end, zustand, lernrate) {
    const H = this.versteckteNeuronen;
    const E = this.embeddingGroesse;
    const V = this.vokabular.length;
    const schritte = [];
    let h = zustand.slice();
    let verlust = 0;

    for (let pos = start; pos < end; pos++) {
      const x = ids[pos];
      const target = ids[pos + 1];
      const step = this.schrittVorwaerts(x, h);
      const p = Math.max(1e-12, step.wahrscheinlichkeiten[target] || 1e-12);
      verlust -= Math.log(p);
      step.target = target;
      schritte.push(step);
      h = step.h;
    }

    if (!schritte.length) return { h, verlust: 0, anzahl: 0 };

    const gEmb = new Map();
    const gWxh = matrix(E, H);
    const gWhh = matrix(H, H);
    const gBh = zeros(H);
    const gWhy = matrix(H, V);
    const gBy = zeros(V);
    let dhWeiter = zeros(H);

    for (let t = schritte.length - 1; t >= 0; t--) {
      const step = schritte[t];
      const p = step.wahrscheinlichkeiten;
      const dLogits = p.slice();
      dLogits[step.target] -= 1;

      for (let j = 0; j < H; j++) {
        for (let v = 0; v < V; v++) gWhy[j][v] += step.h[j] * dLogits[v];
      }
      for (let v = 0; v < V; v++) gBy[v] += dLogits[v];

      const dh = dhWeiter.slice();
      for (let j = 0; j < H; j++) {
        for (let v = 0; v < V; v++) dh[j] += this.gewichteAusgabe[j][v] * dLogits[v];
      }

      const dz = new Array(H);
      for (let j = 0; j < H; j++) {
        dz[j] = dh[j] * (1 - step.h[j] * step.h[j]);
        gBh[j] += dz[j];
      }

      const embedding = this.embeddings[step.tokenId];
      let ge = gEmb.get(step.tokenId);
      if (!ge) {
        ge = zeros(E);
        gEmb.set(step.tokenId, ge);
      }

      for (let i = 0; i < E; i++) {
        let grad = 0;
        for (let j = 0; j < H; j++) {
          gWxh[i][j] += embedding[i] * dz[j];
          grad += this.gewichteEingabe[i][j] * dz[j];
        }
        ge[i] += grad;
      }

      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) gWhh[i][j] += step.hVorher[i] * dz[j];
      }

      const dhVorher = zeros(H);
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < H; j++) dhVorher[i] += this.gewichteRekurrenz[i][j] * dz[j];
      }
      dhWeiter = dhVorher;
    }

    const factor = lernrate / schritte.length;
    const update = (parameter, gradient) => {
      for (let i = 0; i < parameter.length; i++) {
        parameter[i] -= factor * this.begrenzeGradient(gradient[i]);
      }
    };
    const update2D = (parameter, gradient) => {
      for (let i = 0; i < parameter.length; i++) update(parameter[i], gradient[i]);
    };

    update2D(this.gewichteEingabe, gWxh);
    update2D(this.gewichteRekurrenz, gWhh);
    update(this.biasVersteckt, gBh);
    update2D(this.gewichteAusgabe, gWhy);
    update(this.biasAusgabe, gBy);
    for (const [id, gradient] of gEmb.entries()) update(this.embeddings[id], gradient);

    return { h, verlust, anzahl: schritte.length };
  }

  trainiereTexte(daten, tokenizer = null, optionen = {}) {
    if (tokenizer && typeof tokenizer.zerlege !== "function") {
      optionen = tokenizer;
      tokenizer = null;
    }
    if (tokenizer) this.tokenizer = tokenizer;

    const gesammelt = this.sammleTrainingsDaten(daten);
    if (!gesammelt.sequenzen.length) {
      throw new Error("Keine Trainingsdaten gefunden. Nutze Text oder JSON mit Frage/Antwort-Feldern.");
    }

    this.bereit = false;
    this.letzterFehler = null;
    this.trainingsPaare = gesammelt.paare;
    this.konversationsModus = gesammelt.paare.length > 0;
    this.trainingsSequenzen = gesammelt.sequenzen;
    this.baueVokabular(gesammelt.sequenzen);
    this.initialisiereGewichte();

    const kodierteSequenzen = gesammelt.sequenzen
      .map(seq => this.kodiereSequenz(seq))
      .filter(ids => ids.length >= 3);

    if (!kodierteSequenzen.length) throw new Error("Die Trainingsdaten konnten nicht in Tokens umgewandelt werden.");
    this.baueNgramModell(kodierteSequenzen);

    const epochen = Math.max(1, Math.min(30, Math.floor(optionen.epochen || 12)));
    const lernrate = Number.isFinite(optionen.lernrate) ? optionen.lernrate : 0.06;
    const maxTokens = Math.max(100, Math.floor(optionen.maxTrainingsTokens || this.maxTrainingsTokens));
    const segmentLaenge = Math.max(4, Math.min(32, Math.floor(optionen.segmentLaenge || this.trainingsSchrittLaenge)));

    this.trainingsBeispiele = 0;
    this.trainierteEpochen = 0;
    this.verlustHistorie = [];

    for (let epoche = 0; epoche < epochen; epoche++) {
      const reihenfolge = shuffle(kodierteSequenzen.slice());
      const zustaende = new Map(reihenfolge.map((ids, index) => [index, {
        ids,
        position: 0,
        hidden: zeros(this.versteckteNeuronen)
      }]));
      let gelernt = 0;
      let verlustGesamt = 0;
      let verlustTokens = 0;
      const aktive = [...zustaende.values()];

      while (gelernt < maxTokens) {
        let etwasGetan = false;
        for (const eintrag of aktive) {
          if (gelernt >= maxTokens) break;
          if (eintrag.position >= eintrag.ids.length - 1) continue;
          etwasGetan = true;

          const start = eintrag.position;
          const ende = Math.min(
            eintrag.ids.length - 1,
            start + segmentLaenge,
            start + (maxTokens - gelernt)
          );
          if (ende <= start) continue;

          const resultat = this.trainiereAbschnitt(
            eintrag.ids,
            start,
            ende,
            eintrag.hidden,
            lernrate
          );
          eintrag.hidden = resultat.h;
          eintrag.position = ende;
          gelernt += resultat.anzahl;
          verlustGesamt += resultat.verlust;
          verlustTokens += resultat.anzahl;
        }
        if (!etwasGetan) break;
      }

      this.trainingsBeispiele += gelernt;
      this.trainierteEpochen = epoche + 1;
      const mittlererVerlust = verlustTokens ? verlustGesamt / verlustTokens : null;
      this.verlustHistorie.push(mittlererVerlust);
      console.log(`RNN-Training: Epoche ${epoche + 1}/${epochen}, ${gelernt} Token-Schritte, Verlust ${mittlererVerlust === null ? "n/a" : mittlererVerlust.toFixed(3)}`);
    }

    this.bereit = true;
    return this.status();
  }

  ausgabeWahrscheinlichkeiten(hidden, temperatur = 1) {
    const H = this.versteckteNeuronen;
    const logits = new Array(this.vokabular.length);
    for (let v = 0; v < this.vokabular.length; v++) {
      let wert = this.biasAusgabe[v];
      for (let j = 0; j < H; j++) wert += hidden[j] * this.gewichteAusgabe[j][v];
      logits[v] = wert;
    }
    return this.softmax(logits, temperatur);
  }

  bewerteGenerierteSequenz(text, prompt, generierteIds = null) {
    const ids = Array.isArray(generierteIds) && generierteIds.length
      ? generierteIds.slice()
      : this.kodiereText(text, false, false);
    if (!ids.length) return -1000;

    const promptIds = this.kodierePrompt(prompt);
    const praefix = this.konversationsModus
      ? [this.bosId, this.userId, ...promptIds, this.aiId]
      : [this.bosId, ...promptIds];
    let hidden = zeros(this.versteckteNeuronen);
    for (const id of praefix) hidden = this.schrittVorwaerts(id, hidden).h;
    const kontext = praefix.slice();
    let logWahrscheinlichkeit = 0;
    let anzahl = 0;

    for (const id of ids) {
      const nn = this.ausgabeWahrscheinlichkeiten(hidden, 1);
      const ngram = this.findeNgramVerteilung(kontext, tokenId => tokenId !== this.padId && tokenId !== this.bosId && tokenId !== this.userId && tokenId !== this.aiId && tokenId !== this.unkId);
      let p = nn[id] || 1e-12;
      if (ngram) {
        p = (1 - ngram.alpha) * p + ngram.alpha * (ngram.probabilities.get(id) || 0);
      }
      logWahrscheinlichkeit += Math.log(Math.max(1e-12, p));
      anzahl++;
      hidden = this.schrittVorwaerts(id, hidden).h;
      kontext.push(id);
    }

    const woerter = this.zerlegeText(text).filter(token => WORD_RE.test(token));
    if (woerter.length < 3) logWahrscheinlichkeit -= 2;
    if (/^[,.;:!?…]/.test(text.trim())) logWahrscheinlichkeit -= 5;
    if (/(?:[.!?…,]){2,}/.test(text)) logWahrscheinlichkeit -= 2;
    for (let i = 1; i < woerter.length; i++) {
      if (woerter[i].toLocaleLowerCase("de-DE") === woerter[i - 1].toLocaleLowerCase("de-DE")) logWahrscheinlichkeit -= 1.5;
    }
    if (/[.!?…]$/.test(text.trim())) logWahrscheinlichkeit += 0.1;
    return logWahrscheinlichkeit / Math.max(1, anzahl);
  }

  sampleNextToken(hidden, optionen = {}, kontextIds = [], imWort = false, generierteIds = kontextIds) {
    const H = this.versteckteNeuronen;
    const V = this.vokabular.length;
    const temperatur = Math.max(0.2, Math.min(1.5, Number(optionen.temperatur ?? 0.62)));
    const logits = new Array(V);

    for (let v = 0; v < V; v++) {
      let value = this.biasAusgabe[v];
      for (let j = 0; j < H; j++) value += hidden[j] * this.gewichteAusgabe[j][v];

      const token = this.vokabular[v];
      if ([SPECIAL.PAD, SPECIAL.BOS, SPECIAL.USER, SPECIAL.AI, SPECIAL.UNK].includes(token)) value = -1e9;
      if (token === SPECIAL.EOS && !optionen.eosErlaubt) value = -1e9;
      if (token === SPECIAL.END_WORD && !imWort) value = -1e9;
      if (this.charAusToken(token) !== null && !imWort) value = -1e9;
      if (token === SPECIAL.WORD && imWort) value = -1e9;
      if (imWort && token !== SPECIAL.END_WORD && this.charAusToken(token) === null) value = -1e9;

      const wiederholungFenster = Math.max(1, Math.floor(optionen.wiederholungFenster || 12));
      if (generierteIds.slice(-wiederholungFenster).includes(v)) value -= Number(optionen.wiederholungsStrafe ?? 0.38);
      logits[v] = value;
    }

    const probs = this.softmax(logits, temperatur);
    const erlaubt = id => Number.isFinite(logits[id]) && logits[id] > -1e8;
    const ngram = this.findeNgramVerteilung(kontextIds, erlaubt);
    if (ngram) {
      for (let id = 0; id < probs.length; id++) {
        const wordProb = ngram.probabilities.get(id) || 0;
        probs[id] = (1 - ngram.alpha) * probs[id] + ngram.alpha * wordProb;
      }
      const sum = probs.reduce((a, b) => a + b, 0) || 1;
      for (let id = 0; id < probs.length; id++) probs[id] /= sum;
    }

    const kandidaten = probs
      .map((p, id) => ({ id, p }))
      .filter(x => Number.isFinite(x.p) && x.p > 0)
      .sort((a, b) => b.p - a.p)
      .slice(0, Math.max(1, Math.floor(optionen.topK || 8)));

    if (!kandidaten.length) return this.eosId;
    const sum = kandidaten.reduce((s, item) => s + item.p, 0) || 1;
    let r = Math.random() * sum;
    for (const item of kandidaten) {
      r -= item.p;
      if (r <= 0) return item.id;
    }
    return kandidaten[0].id;
  }
    formatiereTokens(tokens) {
    const teile = [];
    let imWort = false;
    let wort = "";

    const setzeWort = () => {
      if (wort) teile.push({ text: wort, art: "wort" });
      wort = "";
      imWort = false;
    };

    for (const token of tokens) {
      if (token === SPECIAL.WORD) {
        setzeWort();
        imWort = true;
        continue;
      }
      if (token === SPECIAL.END_WORD) {
        setzeWort();
        continue;
      }
      if (imWort) {
        const char = this.charAusToken(token);
        if (char !== null) wort += char;
        continue;
      }
      if ([SPECIAL.PAD, SPECIAL.BOS, SPECIAL.EOS, SPECIAL.UNK, SPECIAL.USER, SPECIAL.AI].includes(token)) continue;
      const char = this.charAusToken(token);
      if (char !== null) continue;
      teile.push({ text: token, art: WORD_RE.test(token) ? "wort" : "zeichen" });
    }
    setzeWort();

    let text = "";
    const keineLueckeDavor = new Set([".", ",", "!", "?", ":", ";", "%", ")", "]", "}", "…", "»", "”", "'"]);
    const keineLueckeDanach = new Set(["(", "[", "{", "«", "„", "“"]);
    for (const teil of teile) {
      if (!text) {
        text = teil.text;
      } else if (keineLueckeDavor.has(teil.text)) {
        text += teil.text;
      } else if (keineLueckeDanach.has(text.slice(-1))) {
        text += teil.text;
      } else {
        text += " " + teil.text;
      }
    }
    return text.replace(/\s+/g, " ").trim();
  }

  generiereKandidaten(prompt, plan = null, optionen = {}) {
    const count = Math.max(1, Math.min(8, Math.floor(optionen.anzahlKandidaten || 4)));
    const maxTokens = Math.max(5, Math.min(180, Math.floor(optionen.maxTokens || 55)));
    const seedText = this.kodierePrompt(prompt);
    const praefix = this.konversationsModus
      ? [this.bosId, this.userId, ...seedText, this.aiId]
      : [this.bosId, ...seedText];

    const result = [];
    for (let kandidat = 0; kandidat < count; kandidat++) {
      let hidden = zeros(this.versteckteNeuronen);
      for (const id of praefix) hidden = this.schrittVorwaerts(id, hidden).h;

      const generierteIds = [];
      let imWort = false;
      let zeichenImWort = 0;
      const minTokens = Math.max(1, Math.floor(optionen.minTokens ?? 5));

      for (let i = 0; i < maxTokens; i++) {
        const sampleOptions = { ...optionen, eosErlaubt: i >= minTokens };
        const next = this.sampleNextToken(
          hidden,
          sampleOptions,
          praefix.concat(generierteIds),
          imWort,
          generierteIds
        );
        if (next === this.eosId) break;
        const token = this.vokabular[next];
        if (!token) break;
        if (token === SPECIAL.WORD) {
          imWort = true;
          zeichenImWort = 0;
        } else if (token === SPECIAL.END_WORD) {
          imWort = false;
        } else if (this.charAusToken(token) !== null && imWort) {
          zeichenImWort++;
        }
        if (imWort && zeichenImWort > 30) break;
        generierteIds.push(next);
        hidden = this.schrittVorwaerts(next, hidden).h;
      }

      const tokens = generierteIds.map(id => this.vokabular[id]);
      const text = this.formatiereTokens(tokens);
      result.push({ text, bewertung: this.bewerteGenerierteSequenz(text, prompt, generierteIds) });
    }
    return result;
  }

  bewerteAntwort(text, prompt) {
    if (!text || !text.trim()) return -1000;
    const tokens = this.zerlegeText(text);
    if (!tokens.length) return -1000;
    const wortTokens = tokens.filter(token => WORD_RE.test(token));
    const unique = new Set(wortTokens);
    let score = unique.size / Math.max(1, wortTokens.length);
    if (wortTokens.length < 3) score -= 1.5;
    if (wortTokens.length >= 5 && wortTokens.length <= 40) score += 0.5;

    const promptWorte = new Set(this.zerlegeText(prompt).map(w => w.toLocaleLowerCase("de-DE")));
    for (const wort of wortTokens) {
      if (promptWorte.has(wort.toLocaleLowerCase("de-DE"))) score += 0.03;
    }

    let gleicheNachbarn = 0;
    for (let i = 1; i < wortTokens.length; i++) {
      if (wortTokens[i].toLocaleLowerCase("de-DE") === wortTokens[i - 1].toLocaleLowerCase("de-DE")) gleicheNachbarn++;
    }
    score -= gleicheNachbarn * 1.25;
    return score;
  }

  antwortGenerieren(prompt, optionen = {}) {
    if (!this.bereit) {
      this.letzterFehler = "Das Modell ist noch nicht trainiert. Rufe zuerst trainiereTexte() auf.";
      return "";
    }
    if (typeof prompt !== "string" || !prompt.trim()) return "";

    try {
      const kandidaten = this.generiereKandidaten(prompt, null, optionen)
        .filter(item => item.text && item.text.trim())
        .sort((a, b) => b.bewertung - a.bewertung);

      if (!kandidaten.length) {
        this.letzterFehler = "Das Netz hat keine Textantwort erzeugt. Trainiere mit mehr vollständigen Texten.";
        return "";
      }

      let answer = kandidaten[0].text.trim();
      answer = answer.replace(/\s+([,.!?;:])/g, "$1").trim();
      if (answer && !/[.!?…]$/.test(answer)) answer += ".";
      if (answer) answer = answer[0].toLocaleUpperCase("de-DE") + answer.slice(1);

      this.letzteAntworten.unshift(answer);
      this.letzteAntworten = this.letzteAntworten.slice(0, 10);
      this.letzterFehler = null;
      this.letzteAntwortAnalyse = {
        modus: "rnn_token_fuer_token",
        eingabe: prompt,
        antwort: answer,
        kandidaten: kandidaten.length
      };
      return answer;
    } catch (error) {
      this.letzterFehler = error.message;
      return "";
    }
  }

  antwortGenerierenAlt(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  generiere(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  findePassendesBeispiel(frage) {
    const normalisiere = value => this.zerlegeText(value).map(x => x.toLowerCase());
    const a = new Set(normalisiere(frage).filter(x => WORD_RE.test(x)));
    let best = null;
    let bestScore = 0;
    for (const pair of this.trainingsPaare) {
      const b = normalisiere(pair.frage).filter(x => WORD_RE.test(x));
      const matches = b.filter(w => a.has(w)).length;
      const score = matches / Math.max(1, new Set(b).size + a.size - matches);
      if (score > bestScore) {
        bestScore = score;
        best = pair;
      }
    }
    return { paar: best, punktzahl: bestScore };
  }

  aehnlichkeit(textA, textB) {
    const tokensA = new Set(this.zerlegeText(textA).map(x => x.toLocaleLowerCase("de-DE")));
    const tokensB = new Set(this.zerlegeText(textB).map(x => x.toLocaleLowerCase("de-DE")));
    const intersection = [...tokensA].filter(x => tokensB.has(x)).length;
    const union = new Set([...tokensA, ...tokensB]).size;
    return union ? intersection / union : 0;
  }

  wichtigeWoerter(text) {
    const stop = new Set(["der", "die", "das", "und", "oder", "ein", "eine", "ist", "sind", "ich", "du", "was", "wie", "wo", "warum", "wieso", "mit", "von", "für", "auf", "in", "am", "an", "zu", "auch", "nicht"]);
    return new Set(this.zerlegeText(text).filter(token => WORD_RE.test(token) && token.length > 2 && !stop.has(token.toLocaleLowerCase("de-DE"))));
  }

  planeAntwort(prompt) {
    const treffer = this.findePassendesBeispiel(prompt);
    return {
      eingabe: prompt,
      kernbegriffe: [...this.wichtigeWoerter(prompt)],
      aehnlicheFrage: treffer.punktzahl >= 0.3 ? treffer.paar?.frage || null : null,
    };
  }

  status() {
    return {
      bereit: this.bereit,
      modell: "Kleines rekurrentes neuronales Sprachmodell (RNN), lokal in JavaScript",
      api: false,
      vokabularGroesse: this.vokabular.length,
      versteckteNeuronen: this.versteckteNeuronen,
      embeddingGroesse: this.embeddingGroesse,
      trainingsBeispiele: this.trainingsBeispiele,
      trainierteEpochen: this.trainierteEpochen,
      konversationsModus: this.konversationsModus,
      trainingsPaare: this.trainingsPaare.length,
      trainingsSequenzen: this.trainingsSequenzen.length,
      letzterVerlust: this.verlustHistorie.length ? this.verlustHistorie[this.verlustHistorie.length - 1] : null,
      letzterFehler: this.letzterFehler
    };
  }

  lerneOrdner(ordner, tokenizer = null, optionen = {}) {
    if (tokenizer && typeof tokenizer.zerlege !== "function") {
      optionen = tokenizer;
      tokenizer = null;
    }
    if (tokenizer) this.tokenizer = tokenizer;
    if (!fs.existsSync(ordner)) throw new Error(`Trainingsordner nicht gefunden: ${ordner}`);

    const daten = [];
    const dateien = fs.readdirSync(ordner, { withFileTypes: true });
    for (const datei of dateien) {
      if (!datei.isFile() || !datei.name.toLowerCase().endsWith(".json")) continue;
      if (["tokenizer.json", "netz-modell.json", "model.json"].includes(datei.name.toLowerCase())) continue;
      try {
        daten.push(JSON.parse(fs.readFileSync(path.join(ordner, datei.name), "utf8")));
      } catch (error) {
        console.warn(`Datei ${datei.name} übersprungen: ${error.message}`);
      }
    }
    if (!daten.length) throw new Error(`Keine gültigen JSON-Trainingsdateien in ${ordner} gefunden.`);
    return this.trainiereTexte(daten, tokenizer, optionen);
  }

  speichern(datei = path.join(__dirname, "modelle", "netz-modell.json")) {
    if (!this.bereit) throw new Error("Es gibt noch kein trainiertes Modell zum Speichern.");
    fs.mkdirSync(path.dirname(datei), { recursive: true });
    const serial = {
      version: 2,
      optionen: {
        maxVokabular: this.maxVokabular,
        embeddingGroesse: this.embeddingGroesse,
        versteckteNeuronen: this.versteckteNeuronen,
        kontextLaenge: this.kontextLaenge,
        trainingsSchrittLaenge: this.trainingsSchrittLaenge,
        maxTrainingsTokens: this.maxTrainingsTokens,
        ngramOrdnung: this.ngramOrdnung
      },
      vokabular: this.vokabular,
      embeddings: this.embeddings,
      gewichteEingabe: this.gewichteEingabe,
      gewichteRekurrenz: this.gewichteRekurrenz,
      biasVersteckt: this.biasVersteckt,
      gewichteAusgabe: this.gewichteAusgabe,
      biasAusgabe: this.biasAusgabe,
      trainingsBeispiele: this.trainingsBeispiele,
      trainierteEpochen: this.trainierteEpochen,
      konversationsModus: this.konversationsModus,
      trainingsPaare: this.trainingsPaare,
      ngramCounts: this.ngramCounts.map(level => [...level.entries()].map(([key, values]) => [key, [...values.entries()]]))
    };
    fs.writeFileSync(datei, JSON.stringify(serial), "utf8");
    return datei;
  }

  laden(datei = path.join(__dirname, "modelle", "netz-modell.json")) {
    const serial = JSON.parse(fs.readFileSync(datei, "utf8"));
    if (serial.version !== 2 || !Array.isArray(serial.vokabular)) {
      throw new Error("Ungültige oder nicht unterstützte Modelldatei.");
    }
    const o = serial.optionen || {};
    this.maxVokabular = o.maxVokabular || this.maxVokabular;
    this.embeddingGroesse = o.embeddingGroesse || this.embeddingGroesse;
    this.versteckteNeuronen = o.versteckteNeuronen || this.versteckteNeuronen;
    this.kontextLaenge = o.kontextLaenge || this.kontextLaenge;
    this.trainingsSchrittLaenge = o.trainingsSchrittLaenge || this.trainingsSchrittLaenge;
    this.maxTrainingsTokens = o.maxTrainingsTokens || this.maxTrainingsTokens;
    this.ngramOrdnung = o.ngramOrdnung || this.ngramOrdnung;

    this.vokabular = serial.vokabular;
    this.tokenZuId = new Map(this.vokabular.map((token, id) => [token, id]));
    this.padId = this.tokenZuId.get(SPECIAL.PAD);
    this.bosId = this.tokenZuId.get(SPECIAL.BOS);
    this.eosId = this.tokenZuId.get(SPECIAL.EOS);
    this.unkId = this.tokenZuId.get(SPECIAL.UNK);
    this.userId = this.tokenZuId.get(SPECIAL.USER);
    this.aiId = this.tokenZuId.get(SPECIAL.AI);
    this.wordId = this.tokenZuId.get(SPECIAL.WORD);
    this.endWordId = this.tokenZuId.get(SPECIAL.END_WORD);

    this.embeddings = serial.embeddings;
    this.gewichteEingabe = serial.gewichteEingabe;
    this.gewichteRekurrenz = serial.gewichteRekurrenz;
    this.biasVersteckt = serial.biasVersteckt;
    this.gewichteAusgabe = serial.gewichteAusgabe;
    this.biasAusgabe = serial.biasAusgabe;
    this.trainingsBeispiele = serial.trainingsBeispiele || 0;
    this.trainierteEpochen = serial.trainierteEpochen || 0;
    this.konversationsModus = !!serial.konversationsModus;
    this.trainingsPaare = Array.isArray(serial.trainingsPaare) ? serial.trainingsPaare : [];
    this.ngramCounts = Array.from({ length: this.ngramOrdnung + 1 }, (_, level) => {
      const entries = serial.ngramCounts?.[level] || [];
      return new Map(entries.map(([key, values]) => [key, new Map(values)]));
    });
    this.bereit = true;
    this.letzterFehler = null;
    this.haeufigeWoerter = new Set(this.vokabular.filter(token => WORD_RE.test(token)));
    this.zeichenTokens = new Set(this.vokabular.map(token => this.charAusToken(token)).filter(Boolean));
    return this.status();
  }
}

module.exports = { NeuronalesNetz };
