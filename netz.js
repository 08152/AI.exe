"use strict";

const fs = require("node:fs");
const path = require("node:path");

const STOPWOERTER = new Set([
  "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer",
  "eines", "einem", "und", "oder", "aber", "ist", "sind", "war",
  "waren", "ich", "du", "er", "sie", "es", "wir", "ihr", "was",
  "wie", "wer", "wo", "wann", "warum", "wieso", "mit", "von", "für",
  "auf", "in", "im", "am", "an", "zu", "zum", "zur", "auch", "nicht",
  "kein", "keine", "bitte", "noch", "schon", "sehr", "hat", "haben",
  "kann", "können", "sich", "mir", "mich", "mein", "meine", "dein",
  "deine", "bin", "bist", "seid", "dass", "als", "bei", "nach",
  "vor", "über", "unter", "aus", "einen", "einem", "einer", "dann"
]);

const VERBOTENE_AUSGABE_TOKENS = new Set([
  "<PAD>",
  "<BOS>",
  "<UNK>",
  "<SEP>",
  "<benutzer>",
  "<ki>"
]);

class NeuronalesNetz {
  constructor(tokenizer = null, optionen = {}) {
    if (tokenizer && typeof tokenizer.zerlege !== "function") {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.maxVokabular = Number.isInteger(optionen.maxVokabular)
      ? Math.max(16, optionen.maxVokabular)
      : 512;

    this.embeddingGroesse = Number.isInteger(optionen.embeddingGroesse)
      ? Math.max(2, optionen.embeddingGroesse)
      : 8;

    this.versteckteNeuronen = Number.isInteger(
      optionen.versteckteNeuronen
    )
      ? Math.max(8, optionen.versteckteNeuronen)
      : 120;

    this.kontextLaenge = Number.isInteger(optionen.kontextLaenge)
      ? Math.max(4, optionen.kontextLaenge)
      : 20;

    this.modellOrdner = path.resolve(
      optionen.modellOrdner ||
      process.env.KI_MODELL_ORDNER ||
      path.join(__dirname, "modelle")
    );

    this.modellDatei = path.resolve(
      optionen.modellDatei ||
      path.join(this.modellOrdner, "netz.json")
    );

    this.tokenizerDatei = path.resolve(
      optionen.tokenizerDatei ||
      path.join(this.modellOrdner, "tokenizer.json")
    );

    this.tokenizer = tokenizer;

    this.vokabular = [];
    this.embeddings = null;
    this.gewichte1 = null;
    this.bias1 = null;
    this.gewichte2 = null;
    this.bias2 = null;

    this.padId = 0;
    this.unkId = 1;
    this.bosId = 2;
    this.eosId = 3;

    this.bereit = false;
    this.konversationsModus = false;

    this.trainingsBeispiele = 0;
    this.trainierteEpochen = 0;

    this.trainingsPaare = [];
    this.trainingsTexte = [];

    this.letzteAntworten = [];
    this.letzteAntwortAnalyse = null;
    this.letzterFehler = null;
    this.zuletztGespeichert = null;

    // Gespeicherte Gewichte und Daten automatisch laden.
    this.ladeModell({ still: true });
  }

  // =====================================================
  // TEXTNORMALISIERUNG
  // =====================================================

  normalisiereText(text) {
    return String(text ?? "")
      .normalize("NFC")
      .toLocaleLowerCase("de-DE")
      .trim()
      .replace(/\s+/g, " ");
  }

  normalisiereVergleich(text) {
    return this.normalisiereText(text)
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  // =====================================================
  // TRAININGSDATEN AUS JSON UND TEXTEN LESEN
  // =====================================================

  extrahiereTexte(daten, ergebnis = []) {
    if (typeof daten === "string") {
      const text = this.normalisiereText(daten);

      if (text) {
        ergebnis.push({
          text,
          istKonversation: false
        });
      }

      return ergebnis;
    }

    if (Array.isArray(daten)) {
      for (const element of daten) {
        this.extrahiereTexte(element, ergebnis);
      }

      return ergebnis;
    }

    if (!daten || typeof daten !== "object") {
      return ergebnis;
    }

    const map = Object.create(null);

    for (const key of Object.keys(daten)) {
      map[key.toLowerCase()] = key;
    }

    const frageFeld = [
      "frage",
      "question",
      "prompt",
      "input",
      "user"
    ].find(key => map[key]);

    const antwortFeld = [
      "antwort",
      "answer",
      "response",
      "completion",
      "output",
      "assistant"
    ].find(key => map[key]);

    if (
      frageFeld &&
      antwortFeld &&
      typeof daten[map[frageFeld]] === "string" &&
      typeof daten[map[antwortFeld]] === "string"
    ) {
      const frage = this.normalisiereText(
        daten[map[frageFeld]]
      );

      const antwort = this.normalisiereText(
        daten[map[antwortFeld]]
      );

      if (frage && antwort) {
        this.trainingsPaare.push({
          frage,
          antwort
        });

        ergebnis.push({
          text: `<benutzer> ${frage} <ki> ${antwort}`,
          istKonversation: true
        });
      }

      return ergebnis;
    }

    for (const wert of Object.values(daten)) {
      this.extrahiereTexte(wert, ergebnis);
    }

    return ergebnis;
  }

  // =====================================================
  // TOKENIZER LADEN UND SYNCHRONISIEREN
  // =====================================================

  ladeTokenizer() {
    if (this.tokenizer) {
      return this.tokenizer;
    }

    const { Tokenizer } = require("./tokenizer.js");

    this.tokenizer = fs.existsSync(this.tokenizerDatei)
      ? Tokenizer.laden(this.tokenizerDatei)
      : new Tokenizer();

    return this.tokenizer;
  }

  synchronisiereTokenizerMitVokabular() {
    if (!this.tokenizer) {
      return;
    }

    // Die gespeicherte Token-Reihenfolge muss exakt zu
    // den geladenen neuronalen Gewichten passen.
    this.tokenizer.idZuToken = this.vokabular.slice();
    this.tokenizer.tokenZuId = new Map();

    for (let i = 0; i < this.vokabular.length; i++) {
      this.tokenizer.tokenZuId.set(
        this.vokabular[i],
        i
      );
    }
  }

  // =====================================================
  // NEURONALE GEWICHTE INITIALISIEREN
  // =====================================================

  initialisiereGewichte() {
    const vocabSize = this.vokabular.length;

    const inputSize =
      this.kontextLaenge * this.embeddingGroesse;

    const random = grenze =>
      (Math.random() * 2 - 1) * grenze;

    const grenze1 = Math.sqrt(
      2 / (inputSize + this.versteckteNeuronen)
    );

    const grenze2 = Math.sqrt(
      2 / (this.versteckteNeuronen + vocabSize)
    );

    this.embeddings = Array.from(
      { length: vocabSize },
      () => Array.from(
        { length: this.embeddingGroesse },
        () => random(0.1)
      )
    );

    this.gewichte1 = Array.from(
      { length: inputSize },
      () => Array.from(
        { length: this.versteckteNeuronen },
        () => random(grenze1)
      )
    );

    this.bias1 = Array(
      this.versteckteNeuronen
    ).fill(0);

    this.gewichte2 = Array.from(
      { length: this.versteckteNeuronen },
      () => Array.from(
        { length: vocabSize },
        () => random(grenze2)
      )
    );

    this.bias2 = Array(vocabSize).fill(0);
  }

  begrenze(wert, minimum, maximum) {
    return Math.max(
      minimum,
      Math.min(maximum, wert)
    );
  }

  // =====================================================
  // VORWÄRTSBERECHNUNG
  // =====================================================

  vorwaerts(kontext) {
    if (!this.embeddings || !this.vokabular.length) {
      throw new Error(
        "Das neuronale Netz ist noch nicht initialisiert."
      );
    }

    const eingabe = new Array(
      this.kontextLaenge * this.embeddingGroesse
    );

    let p = 0;

    for (
      let position = 0;
      position < this.kontextLaenge;
      position++
    ) {
      let id = kontext[position];

      if (
        !Number.isInteger(id) ||
        id < 0 ||
        id >= this.vokabular.length
      ) {
        id = this.unkId;
      }

      const embedding =
        this.embeddings[id] ||
        this.embeddings[this.unkId] ||
        this.embeddings[0];

      for (
        let d = 0;
        d < this.embeddingGroesse;
        d++
      ) {
        eingabe[p++] = embedding[d] || 0;
      }
    }

    const versteckt = new Array(
      this.versteckteNeuronen
    ).fill(0);

    for (
      let h = 0;
      h < this.versteckteNeuronen;
      h++
    ) {
      let summe = this.bias1[h];

      for (let i = 0; i < eingabe.length; i++) {
        summe += eingabe[i] * this.gewichte1[i][h];
      }

      versteckt[h] = Math.tanh(summe);
    }

    const logits = new Array(
      this.vokabular.length
    ).fill(0);

    for (
      let v = 0;
      v < this.vokabular.length;
      v++
    ) {
      let summe = this.bias2[v];

      for (
        let h = 0;
        h < this.versteckteNeuronen;
        h++
      ) {
        summe +=
          versteckt[h] * this.gewichte2[h][v];
      }

      logits[v] = summe;
    }

    // Numerisch stabile Softmax-Berechnung.
    let maximum = -Infinity;

    for (const wert of logits) {
      if (wert > maximum) {
        maximum = wert;
      }
    }

    const exponenten = logits.map(wert =>
      Math.exp(Math.max(-60, wert - maximum))
    );

    const gesamt = exponenten.reduce(
      (summe, wert) => summe + wert,
      0
    ) || 1;

    const wahrscheinlichkeiten = exponenten.map(
      wert => wert / gesamt
    );

    return {
      eingabe,
      versteckt,
      logits,
      wahrscheinlichkeiten
    };
  }

  // =====================================================
  // RÜCKWÄRTSBERECHNUNG / LERNEN
  // =====================================================

  trainiereBeispiel(kontext, ziel, lernrate) {
    const ergebnis = this.vorwaerts(kontext);

    const gradAusgabe =
      ergebnis.wahrscheinlichkeiten.slice();

    gradAusgabe[ziel] -= 1;

    const gradVersteckt = new Array(
      this.versteckteNeuronen
    ).fill(0);

    for (
      let h = 0;
      h < this.versteckteNeuronen;
      h++
    ) {
      let summe = 0;

      for (
        let v = 0;
        v < this.vokabular.length;
        v++
      ) {
        summe +=
          this.gewichte2[h][v] *
          gradAusgabe[v];
      }

      gradVersteckt[h] = summe;
    }

    const gradVorAktivierung =
      gradVersteckt.map((wert, h) =>
        wert * (
          1 -
          ergebnis.versteckt[h] *
          ergebnis.versteckt[h]
        )
      );

    const gradEingabe = new Array(
      ergebnis.eingabe.length
    ).fill(0);

    // Gradient vor der Änderung der Gewichte berechnen.
    for (
      let i = 0;
      i < ergebnis.eingabe.length;
      i++
    ) {
      let summe = 0;

      for (
        let h = 0;
        h < this.versteckteNeuronen;
        h++
      ) {
        summe +=
          this.gewichte1[i][h] *
          gradVorAktivierung[h];
      }

      gradEingabe[i] = this.begrenze(
        summe,
        -5,
        5
      );
    }

    // Ausgabeschicht aktualisieren.
    for (
      let v = 0;
      v < this.vokabular.length;
      v++
    ) {
      const fehler = this.begrenze(
        gradAusgabe[v],
        -5,
        5
      );

      this.bias2[v] -= lernrate * fehler;

      for (
        let h = 0;
        h < this.versteckteNeuronen;
        h++
      ) {
        this.gewichte2[h][v] -=
          lernrate *
          ergebnis.versteckt[h] *
          fehler;
      }
    }

    // Verborgene Schicht aktualisieren.
    for (
      let h = 0;
      h < this.versteckteNeuronen;
      h++
    ) {
      const fehler = gradVorAktivierung[h];

      this.bias1[h] -= lernrate * fehler;

      for (
        let i = 0;
        i < ergebnis.eingabe.length;
        i++
      ) {
        this.gewichte1[i][h] -=
          lernrate *
          ergebnis.eingabe[i] *
          fehler;
      }
    }

    // Wort-Embeddings aktualisieren.
    for (
      let position = 0;
      position < this.kontextLaenge;
      position++
    ) {
      let id = kontext[position];

      if (
        !Number.isInteger(id) ||
        id < 0 ||
        id >= this.vokabular.length
      ) {
        id = this.unkId;
      }

      for (
        let d = 0;
        d < this.embeddingGroesse;
        d++
      ) {
        const index =
          position * this.embeddingGroesse + d;

        this.embeddings[id][d] -=
          lernrate * gradEingabe[index];
      }
    }
  }

  // =====================================================
  // DOPPELTE TRAININGSDATEN ENTFERNEN
  // =====================================================

  eindeutigeTexte(texte) {
    const gesehen = new Set();
    const ausgabe = [];

    for (const element of texte) {
      if (
        !element ||
        typeof element.text !== "string" ||
        !element.text.trim()
      ) {
        continue;
      }

      const text = this.normalisiereText(element.text);
      const istKonversation = Boolean(
        element.istKonversation
      );

      const key =
        `${istKonversation ? "1" : "0"}\n${text}`;

      if (gesehen.has(key)) {
        continue;
      }

      gesehen.add(key);

      ausgabe.push({
        text,
        istKonversation
      });
    }

    return ausgabe;
  }

  eindeutigePaare(paare) {
    const gesehen = new Set();
    const ausgabe = [];

    for (const paar of paare) {
      if (
        !paar ||
        typeof paar.frage !== "string" ||
        typeof paar.antwort !== "string"
      ) {
        continue;
      }

      const frage = this.normalisiereText(paar.frage);
      const antwort = this.normalisiereText(paar.antwort);

      if (!frage || !antwort) {
        continue;
      }

      const key = `${frage}\n${antwort}`;

      if (gesehen.has(key)) {
        continue;
      }

      gesehen.add(key);
      ausgabe.push({ frage, antwort });
    }

    return ausgabe;
  }

  // =====================================================
  // SPRACHMODELL TRAINIEREN
  // =====================================================

  trainiereTexte(daten, tokenizer = null, optionen = {}) {
    if (
      tokenizer &&
      typeof tokenizer.zerlege !== "function"
    ) {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.bereit = false;
    this.letzterFehler = null;

    const alteTexte = Array.isArray(this.trainingsTexte)
      ? this.trainingsTexte.slice()
      : [];

    const altePaare = Array.isArray(this.trainingsPaare)
      ? this.trainingsPaare.slice()
      : [];

    this.trainingsPaare = [];

    const neueTexte = this.extrahiereTexte(daten, []);
    const neuePaare = this.trainingsPaare.slice();

    // Standardmäßig bleibt das bisherige Wissen erhalten.
    if (optionen.ersetzeVorwissen) {
      this.trainingsTexte =
        this.eindeutigeTexte(neueTexte);

      this.trainingsPaare =
        this.eindeutigePaare(neuePaare);
    } else {
      this.trainingsTexte = this.eindeutigeTexte([
        ...alteTexte,
        ...neueTexte
      ]);

      this.trainingsPaare = this.eindeutigePaare([
        ...altePaare,
        ...neuePaare
      ]);
    }

    if (this.trainingsTexte.length === 0) {
      this.trainingsPaare = altePaare;
      this.trainingsTexte = alteTexte;

      throw new Error(
        "Keine Trainings-Texte gefunden."
      );
    }

    this.tokenizer =
      tokenizer ||
      this.tokenizer ||
      this.ladeTokenizer();

    this.tokenizer.lerneTexte(
      this.trainingsTexte.map(element => element.text)
    );

    this.konversationsModus =
      this.trainingsTexte.some(
        element => element.istKonversation
      );

    this.vokabular =
      this.tokenizer.idZuToken.slice(
        0,
        this.maxVokabular
      );

    if (this.vokabular.length < 5) {
      throw new Error("Das Vokabular ist zu klein.");
    }

    const idVon = (token, fallback) => {
      const id = this.vokabular.indexOf(token);
      return id >= 0 ? id : fallback;
    };

    this.padId = idVon("<PAD>", 0);
    this.unkId = idVon("<UNK>", 1);
    this.bosId = idVon("<BOS>", 2);
    this.eosId = idVon("<EOS>", 3);

    this.initialisiereGewichte();

    const sequenzen = [];

    for (const element of this.trainingsTexte) {
      const ids = this.tokenizer
        .zerlege(element.text)
        .map(token => {
          const id = this.tokenizer.tokenZuId.get(token);

          return Number.isInteger(id) &&
            id >= 0 &&
            id < this.vokabular.length
            ? id
            : this.unkId;
        });

      if (ids.length > 0) {
        sequenzen.push([...ids, this.eosId]);
      }
    }

    const maxBeispiele = Math.max(
      50,
      Math.floor(
        optionen.maxTrainingsBeispiele || 1200
      )
    );

    const anzahlZiele = sequenzen.reduce(
      (summe, sequenz) => summe + sequenz.length,
      0
    );

    const schritt = Math.max(
      1,
      Math.ceil(anzahlZiele / maxBeispiele)
    );

    const beispiele = [];
    let globalePosition = 0;

    for (const sequenz of sequenzen) {
      let kontext = Array(
        this.kontextLaenge
      ).fill(this.bosId);

      for (const ziel of sequenz) {
        if (globalePosition % schritt === 0) {
          beispiele.push({
            kontext: kontext.slice(),
            ziel
          });
        }

        kontext = kontext.slice(1).concat(ziel);
        globalePosition++;
      }
    }

    if (beispiele.length === 0) {
      throw new Error(
        "Keine Trainingsbeispiele erstellt."
      );
    }

    const epochen = Math.max(
      1,
      Math.min(
        30,
        Math.floor(optionen.epochen || 5)
      )
    );

    const lernrate = Number.isFinite(optionen.lernrate)
      ? this.begrenze(optionen.lernrate, 0.0001, 0.1)
      : 0.012;

    for (
      let epoche = 0;
      epoche < epochen;
      epoche++
    ) {
      // Beispiele mischen, damit nicht immer in
      // der identischen Reihenfolge trainiert wird.
      for (
        let i = beispiele.length - 1;
        i > 0;
        i--
      ) {
        const j = Math.floor(
          Math.random() * (i + 1)
        );

        [
          beispiele[i],
          beispiele[j]
        ] = [
          beispiele[j],
          beispiele[i]
        ];
      }

      for (const beispiel of beispiele) {
        this.trainiereBeispiel(
          beispiel.kontext,
          beispiel.ziel,
          lernrate
        );
      }

      console.log(
        `Sprachmodell: Epoche ${epoche + 1}/${epochen}`
      );
    }

    this.trainingsBeispiele = beispiele.length;
    this.trainierteEpochen = epochen;
    this.bereit = true;

    // Wichtig: Gewichte UND Trainingswissen speichern.
    this.speichereTokenizer();
    this.speichernModell();

    return this.status();
  }

  // =====================================================
  // FRAGEÄHNLICHKEIT UND ANTWORTPLANUNG
  // =====================================================

  wichtigeWoerter(text) {
    const tokens = this.tokenizer
      ? this.tokenizer.zerlege(
          this.normalisiereVergleich(text)
        )
      : (
          this.normalisiereVergleich(text)
            .match(/[\p{L}\p{N}]+/gu) || []
        );

    const wichtig = tokens.filter(token =>
      /[\p{L}\p{N}]/u.test(token) &&
      !token.startsWith("<") &&
      !STOPWOERTER.has(token) &&
      (token.length > 2 || /^\d+$/u.test(token))
    );

    return new Set(
      wichtig.length
        ? wichtig
        : tokens.filter(token => !token.startsWith("<"))
    );
  }

  aehnlichkeit(textA, textB) {
    const normA = this.normalisiereVergleich(textA);
    const normB = this.normalisiereVergleich(textB);

    if (!normA || !normB) {
      return 0;
    }

    if (normA === normB) {
      return 1;
    }

    const a = this.wichtigeWoerter(normA);
    const b = this.wichtigeWoerter(normB);

    if (!a.size || !b.size) {
      return 0;
    }

    let gemeinsam = 0;

    for (const wort of a) {
      if (b.has(wort)) {
        gemeinsam++;
      }
    }

    const jaccard =
      gemeinsam / new Set([...a, ...b]).size;

    const abdeckung = gemeinsam / a.size;

    return Math.min(
      1,
      0.6 * jaccard + 0.4 * abdeckung
    );
  }

  findePassendeBeispiele(frage, grenze = 0) {
    const treffer = [];
    const normalisiert =
      this.normalisiereVergleich(frage);

    for (const paar of this.trainingsPaare) {
      const punktzahl =
        normalisiert ===
        this.normalisiereVergleich(paar.frage)
          ? 1
          : this.aehnlichkeit(frage, paar.frage);

      if (punktzahl >= grenze) {
        treffer.push({
          paar,
          punktzahl
        });
      }
    }

    return treffer.sort(
      (a, b) => b.punktzahl - a.punktzahl
    );
  }

  findePassendesBeispiel(frage) {
    const treffer =
      this.findePassendeBeispiele(frage, 0);

    return treffer.length
      ? {
          paar: treffer[0].paar,
          punktzahl: treffer[0].punktzahl
        }
      : {
          paar: null,
          punktzahl: 0
        };
  }

  planeAntwort(prompt) {
    const treffer =
      this.findePassendesBeispiel(prompt);

    const kernbegriffe = new Set(
      this.wichtigeWoerter(prompt)
    );

    if (
      treffer.paar &&
      treffer.punktzahl >= 0.25
    ) {
      for (const wort of this.wichtigeWoerter(
        treffer.paar.antwort
      )) {
        kernbegriffe.add(wort);
      }
    }

    return {
      eingabe: prompt,
      ziel: "Eine passende gelernte Antwort auswählen oder Tokens berechnen.",
      kernbegriffe: [...kernbegriffe],
      beispielAntwort:
        treffer.paar && treffer.punktzahl >= 0.25
          ? treffer.paar.antwort
          : null,
      beispielFrage:
        treffer.paar && treffer.punktzahl >= 0.25
          ? treffer.paar.frage
          : null,
      relevanz: treffer.punktzahl
    };
  }

  // =====================================================
  // TEXT IN TOKEN-KONTEXT UMWANDELN
  // =====================================================

  kontextAusText(text) {
    let kontext = Array(
      this.kontextLaenge
    ).fill(this.bosId);

    for (const token of this.tokenizer.zerlege(text)) {
      const gefunden =
        this.tokenizer.tokenZuId.get(token);

      const id =
        Number.isInteger(gefunden) &&
        gefunden >= 0 &&
        gefunden < this.vokabular.length
          ? gefunden
          : this.unkId;

      kontext = kontext.slice(1).concat(id);
    }

    return kontext;
  }

  idFuerToken(token) {
    const id = this.tokenizer.tokenZuId.get(token);

    return Number.isInteger(id) &&
      id >= 0 &&
      id < this.vokabular.length
      ? id
      : this.unkId;
  }

  // =====================================================
  // WAHRSCHEINLICHKEIT EINER GANZEN ANTWORT
  // =====================================================

  berechneAntwortWahrscheinlichkeit(frage, antwort) {
    if (
      !this.bereit ||
      !this.embeddings ||
      !this.tokenizer
    ) {
      return {
        wahrscheinlichkeit: 0,
        logWahrscheinlichkeit: -Infinity,
        durchschnittlicheTokenWahrscheinlichkeit: 0,
        tokenAnzahl: 0
      };
    }

    const eingabe = this.konversationsModus
      ? `<benutzer> ${this.normalisiereText(frage)} <ki>`
      : this.normalisiereText(frage);

    let kontext = this.kontextAusText(eingabe);

    const antwortTokens = this.tokenizer.zerlege(
      this.normalisiereText(antwort)
    );

    let logWahrscheinlichkeit = 0;
    let tokenAnzahl = 0;

    for (const token of antwortTokens) {
      const id = this.idFuerToken(token);
      const ergebnis = this.vorwaerts(kontext);

      const p = Math.max(
        1e-12,
        ergebnis.wahrscheinlichkeiten[id] || 0
      );

      logWahrscheinlichkeit += Math.log(p);
      tokenAnzahl++;

      kontext = kontext.slice(1).concat(id);
    }

    // Auch die Wahrscheinlichkeit für das Ende
    // des Antwortsatzes in die Rechnung einbeziehen.
    const ende = this.vorwaerts(kontext);

    const pEnde = Math.max(
      1e-12,
      ende.wahrscheinlichkeiten[this.eosId] || 0
    );

    logWahrscheinlichkeit += Math.log(pEnde);
    tokenAnzahl++;

    const durchschnitt = Math.exp(
      logWahrscheinlichkeit /
      Math.max(1, tokenAnzahl)
    );

    return {
      wahrscheinlichkeit:
        logWahrscheinlichkeit < -745
          ? 0
          : Math.exp(logWahrscheinlichkeit),

      logWahrscheinlichkeit,

      durchschnittlicheTokenWahrscheinlichkeit:
        durchschnitt,

      tokenAnzahl
    };
  }

  wahrscheinlichsteNaechsteWoerter(
    frage,
    anzahl = 10
  ) {
    if (!this.bereit || !this.tokenizer) {
      return [];
    }

    const eingabe = this.konversationsModus
      ? `<benutzer> ${this.normalisiereText(frage)} <ki>`
      : this.normalisiereText(frage);

    const ergebnis = this.vorwaerts(
      this.kontextAusText(eingabe)
    );

    return ergebnis.wahrscheinlichkeiten
      .map((p, id) => ({
        wort: this.vokabular[id],
        wahrscheinlichkeit: p,
        id
      }))
      .filter(element =>
        element.wort &&
        !VERBOTENE_AUSGABE_TOKENS.has(element.wort) &&
        element.wort !== "<EOS>"
      )
      .sort(
        (a, b) =>
          b.wahrscheinlichkeit - a.wahrscheinlichkeit
      )
      .slice(
        0,
        Math.max(1, Math.min(50, Math.floor(anzahl)))
      );
  }

  // =====================================================
  // NÄCHSTES TOKEN AUSWÄHLEN
  // =====================================================

  waehleNaechstesToken(
    kontext,
    temperatur = 0.65,
    topK = 5
  ) {
    const ergebnis = this.vorwaerts(kontext);

    const temp = this.begrenze(
      Number(temperatur) || 0.65,
      0.1,
      2
    );

    const kandidaten = ergebnis.logits
      .map((wert, id) => ({
        id,
        wert: wert / temp,
        token: this.vokabular[id]
      }))
      .filter(element =>
        element.token &&
        !VERBOTENE_AUSGABE_TOKENS.has(element.token)
      )
      .sort(
        (a, b) => b.wert - a.wert
      )
      .slice(
        0,
        Math.max(1, Math.floor(topK || 5))
      );

    if (!kandidaten.length) {
      return this.eosId;
    }

    const maximum = kandidaten[0].wert;

    const gewichte = kandidaten.map(element =>
      Math.exp(
        Math.max(-60, element.wert - maximum)
      )
    );

    const gesamt = gewichte.reduce(
      (summe, wert) => summe + wert,
      0
    ) || 1;

    let zufall = Math.random() * gesamt;

    for (let i = 0; i < kandidaten.length; i++) {
      zufall -= gewichte[i];

      if (zufall <= 0) {
        return kandidaten[i].id;
      }
    }

    return kandidaten[0].id;
  }

  // =====================================================
  // TOKENFOLGE FORMATIEREN
  // =====================================================

  formatiere(tokens) {
    let text = tokens.join(" ")
      .replace(/\s+([.,!?;:%)\]}»])/g, "$1")
      .replace(/([([{«])\s+/g, "$1")
      .replace(/\s+/g, " ")
      .trim();

    if (text) {
      text =
        text.charAt(0).toLocaleUpperCase("de-DE") +
        text.slice(1);
    }

    return text;
  }

  // =====================================================
  // FREIE TOKEN-GENERIERUNG
  // =====================================================

  generiereAusNetz(prompt, optionen = {}) {
    const maxTokens = Math.max(
      1,
      Math.min(
        80,
        Math.floor(optionen.maxTokens || 30)
      )
    );

    const temperatur = Number.isFinite(
      optionen.temperatur
    )
      ? optionen.temperatur
      : 0.55;

    const topK = Number.isFinite(optionen.topK)
      ? optionen.topK
      : 4;

    const eingabe = this.konversationsModus
      ? `<benutzer> ${this.normalisiereText(prompt)} <ki>`
      : this.normalisiereText(prompt);

    let kontext = this.kontextAusText(eingabe);

    const erzeugte = [];
    const woerter = new Map();

    for (let i = 0; i < maxTokens; i++) {
      const id = this.waehleNaechstesToken(
        kontext,
        temperatur,
        topK
      );

      if (id === this.eosId) {
        break;
      }

      const token = this.vokabular[id];

      if (
        !token ||
        VERBOTENE_AUSGABE_TOKENS.has(token) ||
        token === "<EOS>"
      ) {
        break;
      }

      const klein = token.toLocaleLowerCase("de-DE");
      const anzahl = woerter.get(klein) || 0;

      if (
        anzahl >= 2 &&
        /[\p{L}\p{N}]/u.test(token)
      ) {
        kontext = kontext.slice(1).concat(id);

        if (anzahl >= 3) {
          break;
        }

        woerter.set(klein, anzahl + 1);
        continue;
      }

      erzeugte.push(token);

      if (/[\p{L}\p{N}]/u.test(token)) {
        woerter.set(klein, anzahl + 1);
      }

      kontext = kontext.slice(1).concat(id);
    }

    return this.formatiere(erzeugte);
  }

  qualitaetAntwort(text, prompt) {
    const tokens = this.tokenizer
      .zerlege(text)
      .filter(token =>
        /[\p{L}\p{N}]/u.test(token) &&
        !token.startsWith("<")
      );

    if (tokens.length < 3) {
      return -10;
    }

    const verschieden =
      new Set(tokens).size / tokens.length;

    let wiederholung = 0;

    for (let i = 1; i < tokens.length; i++) {
      if (tokens[i] === tokens[i - 1]) {
        wiederholung++;
      }
    }

    const relevante = this.wichtigeWoerter(text);
    const frageWoerter = this.wichtigeWoerter(prompt);

    let gemeinsam = 0;

    for (const wort of relevante) {
      if (frageWoerter.has(wort)) {
        gemeinsam++;
      }
    }

    return verschieden * 2 -
      wiederholung * 0.8 +
      Math.min(1, gemeinsam * 0.2);
  }

  generiereKandidaten(
    prompt,
    plan = this.planeAntwort(prompt),
    optionen = {}
  ) {
    const anzahl = Math.max(
      1,
      Math.min(
        6,
        Math.floor(optionen.anzahlKandidaten || 3)
      )
    );

    const kandidaten = [];

    for (let i = 0; i < anzahl; i++) {
      const text = this.generiereAusNetz(
        prompt,
        {
          temperatur:
            (optionen.temperatur ?? 0.55) +
            i * 0.08,
          topK: optionen.topK ?? 4,
          maxTokens: optionen.maxTokens ?? 30
        }
      );

      if (!text) {
        continue;
      }

      const p = this.berechneAntwortWahrscheinlichkeit(
        prompt,
        text
      );

      kandidaten.push({
        text,

        bewertung:
          this.qualitaetAntwort(text, prompt) +
          0.15 * Math.log(
            Math.max(
              1e-12,
              p.durchschnittlicheTokenWahrscheinlichkeit
            )
          ),

        logWahrscheinlichkeit:
          p.logWahrscheinlichkeit
      });
    }

    return kandidaten.sort(
      (a, b) => b.bewertung - a.bewertung
    );
  }

  // =====================================================
  // ANTWORT BERECHNEN
  // =====================================================

  antwortGenerieren(prompt, optionen = {}) {
    if (!this.bereit) {
      return (
        "Mein neuronales Sprachmodell ist noch nicht trainiert. " +
        "Bitte überprüfe deine Trainingsdaten."
      );
    }

    if (
      typeof prompt !== "string" ||
      !prompt.trim()
    ) {
      return "Bitte gib eine Nachricht ein.";
    }

    const normFrage =
      this.normalisiereVergleich(prompt);

    const treffer =
      this.findePassendeBeispiele(prompt, 0.20);

    const exakte = treffer.filter(element =>
      this.normalisiereVergleich(
        element.paar.frage
      ) === normFrage
    );

    let kandidatenTreffer = exakte.length
      ? exakte
      : treffer.filter(element =>
          element.punktzahl >=
          (optionen.minRelevanz ?? 0.30)
        );

    if (kandidatenTreffer.length) {
      const antwortMap = new Map();

      // Doppelte Antworten zusammenfassen.
      for (const trefferElement of kandidatenTreffer) {
        const antwort = trefferElement.paar.antwort;
        const key = this.normalisiereVergleich(antwort);
        const alt = antwortMap.get(key);

        if (
          !alt ||
          trefferElement.punktzahl > alt.relevanz
        ) {
          antwortMap.set(key, {
            text: antwort,
            relevanz: trefferElement.punktzahl,
            frage: trefferElement.paar.frage
          });
        }
      }

      const auswahl = [...antwortMap.values()];

      // Jede Antwort anhand der tatsächlichen
      // Tokenwahrscheinlichkeiten bewerten.
      for (const kandidat of auswahl) {
        const p = this.berechneAntwortWahrscheinlichkeit(
          prompt,
          kandidat.text
        );

        kandidat.wahrscheinlichkeit =
          p.wahrscheinlichkeit;

        kandidat.logWahrscheinlichkeit =
          p.logWahrscheinlichkeit;

        kandidat.tokenQualitaet = Math.log(
          Math.max(
            1e-12,
            p.durchschnittlicheTokenWahrscheinlichkeit
          )
        );

        kandidat.bewertung =
          2.2 * kandidat.relevanz +
          0.15 * kandidat.tokenQualitaet;
      }

      auswahl.sort(
        (a, b) => b.bewertung - a.bewertung
      );

      const beste = auswahl[0];

      this.letzteAntwortAnalyse = {
        modus: exakte.length
          ? "exakter_trainingsfund"
          : "aehnliche_trainingsfrage",
        frage: prompt,
        antwort: beste.text,
        relevanz: beste.relevanz,
        bewertung: beste.bewertung,
        wahrscheinlichkeit: beste.wahrscheinlichkeit,
        logWahrscheinlichkeit: beste.logWahrscheinlichkeit,
        anzahlKandidaten: auswahl.length
      };

      return beste.text;
    }

    // Wenn es Trainingspaare gibt, aber keines passt,
    // lieber Wissenslücken zugeben als Zufall als Wissen auszugeben.
    if (
      this.trainingsPaare.length &&
      optionen.erlaubeFreieGenerierung !== true
    ) {
      this.letzteAntwortAnalyse = {
        modus: "keine_passende_trainingsfrage",
        frage: prompt,
        antwort: null,
        anzahlKandidaten: 0
      };

      return (
        "Dazu habe ich noch keine passende Trainingsantwort gelernt. " +
        "Bringe mir dieses Thema mit einem Frage-Antwort-Beispiel bei."
      );
    }

    const kandidaten = this.generiereKandidaten(
      prompt,
      this.planeAntwort(prompt),
      optionen
    );

    const beste = kandidaten[0];

    if (!beste) {
      return "Ich konnte noch keine zusammenhängende Antwort berechnen.";
    }

    this.letzteAntwortAnalyse = {
      modus: "freie_token_generierung",
      frage: prompt,
      antwort: beste.text,
      bewertung: beste.bewertung,
      logWahrscheinlichkeit: beste.logWahrscheinlichkeit,
      anzahlKandidaten: kandidaten.length
    };

    return beste.text;
  }

  antwortGenerierenAlt(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  generiere(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  // =====================================================
  // NEUES WISSEN LERNEN UND SOFORT SPEICHERN
  // =====================================================

  lerneAntwort(frage, antwort, optionen = {}) {
    if (
      typeof frage !== "string" ||
      !frage.trim() ||
      typeof antwort !== "string" ||
      !antwort.trim()
    ) {
      throw new TypeError(
        "Frage und Antwort müssen nicht-leere Texte sein."
      );
    }

    return this.trainiereTexte(
      [{
        frage: frage.trim(),
        antwort: antwort.trim()
      }],
      null,
      {
        ...optionen,
        epochen: optionen.epochen ?? 3
      }
    );
  }

  // =====================================================
  // TOKENIZER DAUERHAFT SPEICHERN
  // =====================================================

  speichereTokenizer() {
    if (
      !this.tokenizer ||
      typeof this.tokenizer.speichern !== "function"
    ) {
      return false;
    }

    try {
      fs.mkdirSync(
        path.dirname(this.tokenizerDatei),
        { recursive: true }
      );

      this.tokenizer.speichern(
        this.tokenizerDatei
      );

      return true;
    } catch (fehler) {
      this.letzterFehler =
        `Tokenizer speichern fehlgeschlagen: ${fehler.message}`;

      console.warn(this.letzterFehler);
      return false;
    }
  }

  // =====================================================
  // MODELL UND GELERNTES WISSEN SPEICHERN
  // =====================================================

  speichernModell() {
    if (
      !this.bereit ||
      !this.embeddings ||
      !this.vokabular.length
    ) {
      return false;
    }

    try {
      fs.mkdirSync(
        path.dirname(this.modellDatei),
        { recursive: true }
      );

      const daten = {
        schemaVersion: 2,
        gespeichertAm: new Date().toISOString(),

        kontextLaenge: this.kontextLaenge,
        embeddingGroesse: this.embeddingGroesse,
        versteckteNeuronen: this.versteckteNeuronen,
        maxVokabular: this.maxVokabular,

        vokabular: this.vokabular,

        padId: this.padId,
        unkId: this.unkId,
        bosId: this.bosId,
        eosId: this.eosId,

        konversationsModus: this.konversationsModus,

        trainingsBeispiele: this.trainingsBeispiele,
        trainierteEpochen: this.trainierteEpochen,

        trainingsTexte: this.trainingsTexte,
        trainingsPaare: this.trainingsPaare,

        embeddings: this.embeddings,
        gewichte1: this.gewichte1,
        bias1: this.bias1,
        gewichte2: this.gewichte2,
        bias2: this.bias2
      };

      const tempDatei =
        `${this.modellDatei}.tmp`;

      fs.writeFileSync(
        tempDatei,
        JSON.stringify(daten),
        "utf8"
      );

      try {
        fs.renameSync(
          tempDatei,
          this.modellDatei
        );
      } catch {
        fs.copyFileSync(
          tempDatei,
          this.modellDatei
        );

        fs.unlinkSync(tempDatei);
      }

      this.zuletztGespeichert = daten.gespeichertAm;

      return true;
    } catch (fehler) {
      this.letzterFehler =
        `Modell speichern fehlgeschlagen: ${fehler.message}`;

      console.error(this.letzterFehler);
      return false;
    }
  }

  // =====================================================
  // GESPEICHERTES MODELL AUTOMATISCH LADEN
  // =====================================================

  ladeModell(optionen = {}) {
    if (!fs.existsSync(this.modellDatei)) {
      return false;
    }

    try {
      const daten = JSON.parse(
        fs.readFileSync(
          this.modellDatei,
          "utf8"
        )
      );

      if (
        !Array.isArray(daten.vokabular) ||
        !Array.isArray(daten.embeddings) ||
        !Array.isArray(daten.gewichte1) ||
        !Array.isArray(daten.gewichte2)
      ) {
        throw new Error(
          "Die Modelldatei enthält keine gültigen Gewichte."
        );
      }

      this.kontextLaenge =
        Number.isInteger(daten.kontextLaenge)
          ? daten.kontextLaenge
          : this.kontextLaenge;

      this.embeddingGroesse =
        Number.isInteger(daten.embeddingGroesse)
          ? daten.embeddingGroesse
          : this.embeddingGroesse;

      this.versteckteNeuronen =
        Number.isInteger(daten.versteckteNeuronen)
          ? daten.versteckteNeuronen
          : this.versteckteNeuronen;

      this.maxVokabular =
        Number.isInteger(daten.maxVokabular)
          ? daten.maxVokabular
          : this.maxVokabular;

      this.tokenizer =
        this.tokenizer || this.ladeTokenizer();

      this.vokabular = daten.vokabular.slice();

      this.synchronisiereTokenizerMitVokabular();

      this.embeddings = daten.embeddings;
      this.gewichte1 = daten.gewichte1;
      this.bias1 = daten.bias1;
      this.gewichte2 = daten.gewichte2;
      this.bias2 = daten.bias2;

      this.padId = Number.isInteger(daten.padId)
        ? daten.padId
        : 0;

      this.unkId = Number.isInteger(daten.unkId)
        ? daten.unkId
        : 1;

      this.bosId = Number.isInteger(daten.bosId)
        ? daten.bosId
        : 2;

      this.eosId = Number.isInteger(daten.eosId)
        ? daten.eosId
        : 3;

      this.konversationsModus =
        Boolean(daten.konversationsModus);

      this.trainingsBeispiele =
        Number(daten.trainingsBeispiele) || 0;

      this.trainierteEpochen =
        Number(daten.trainierteEpochen) || 0;

      this.trainingsTexte =
        Array.isArray(daten.trainingsTexte)
          ? daten.trainingsTexte
          : [];

      this.trainingsPaare = this.eindeutigePaare(
        Array.isArray(daten.trainingsPaare)
          ? daten.trainingsPaare
          : []
      );

      this.zuletztGespeichert =
        daten.gespeichertAm || null;

      const vocab = this.vokabular.length;

      const gueltig =
        this.embeddings.length === vocab &&
        this.embeddings.every(
          x =>
            Array.isArray(x) &&
            x.length === this.embeddingGroesse
        ) &&
        this.gewichte1.length ===
          this.kontextLaenge * this.embeddingGroesse &&
        this.gewichte1.every(
          x =>
            Array.isArray(x) &&
            x.length === this.versteckteNeuronen
        ) &&
        Array.isArray(this.bias1) &&
        this.bias1.length === this.versteckteNeuronen &&
        this.gewichte2.length === this.versteckteNeuronen &&
        this.gewichte2.every(
          x =>
            Array.isArray(x) &&
            x.length === vocab
        ) &&
        Array.isArray(this.bias2) &&
        this.bias2.length === vocab;

      if (!gueltig) {
        throw new Error(
          "Die gespeicherten Gewichtsgrößen passen nicht zusammen."
        );
      }

      this.bereit = true;

      if (!optionen.still) {
        console.log(
          `[KI] Gespeichertes Modell geladen: ${this.modellDatei}`
        );
      }

      return true;
    } catch (fehler) {
      this.bereit = false;

      this.letzterFehler =
        `Modell laden fehlgeschlagen: ${fehler.message}`;

      if (!optionen.still) {
        console.warn(this.letzterFehler);
      }

      return false;
    }
  }

  // =====================================================
  // STATUS
  // =====================================================

  status() {
    return {
      bereit: this.bereit,
      modell: "Kleines neuronales Sprachmodell mit dauerhaftem Speicher",

      versteckteNeuronen: this.versteckteNeuronen,
      vokabularGroesse: this.vokabular.length,

      trainingsBeispiele: this.trainingsBeispiele,
      trainierteEpochen: this.trainierteEpochen,
      kontextLaenge: this.kontextLaenge,

      konversationsModus: this.konversationsModus,
      trainingsPaare: this.trainingsPaare.length,
      trainingsTexte: this.trainingsTexte.length,

      gespeichert: Boolean(this.zuletztGespeichert),
      zuletztGespeichert: this.zuletztGespeichert,
      modellDatei: this.modellDatei,

      letzterFehler: this.letzterFehler
    };
  }

  // =====================================================
  // TRAININGSORDNER LADEN
  // =====================================================

  lerneOrdner(
    ordner,
    tokenizer = null,
    optionen = {}
  ) {
    if (
      tokenizer &&
      typeof tokenizer.zerlege !== "function"
    ) {
      optionen = tokenizer;
      tokenizer = null;
    }

    if (!fs.existsSync(ordner)) {
      throw new Error(
        `Trainingsordner nicht gefunden: ${ordner}`
      );
    }

    const daten = [];

    const dateien = fs.readdirSync(
      ordner,
      { withFileTypes: true }
    );

    for (const datei of dateien) {
      if (
        !datei.isFile() ||
        !/\.(json|jsonl)$/i.test(datei.name)
      ) {
        continue;
      }

      if ([
        "tokenizer.json",
        "netz.json",
        "model.json"
      ].includes(datei.name.toLowerCase())) {
        continue;
      }

      const dateipfad = path.join(
        ordner,
        datei.name
      );

      try {
        const raw = fs.readFileSync(
          dateipfad,
          "utf8"
        );

        if (datei.name.toLowerCase().endsWith(".jsonl")) {
          for (const zeile of raw.split(/\r?\n/).filter(Boolean)) {
            try {
              daten.push(JSON.parse(zeile));
            } catch (fehler) {
              console.warn(
                `Zeile in ${datei.name} übersprungen: ${fehler.message}`
              );
            }
          }
        } else {
          daten.push(JSON.parse(raw));
        }
      } catch (fehler) {
        console.warn(
          `Trainingsdatei ${datei.name} übersprungen: ${fehler.message}`
        );
      }
    }

    if (!daten.length) {
      throw new Error(
        `Keine passenden JSON-Trainingsdateien in ${ordner} gefunden.`
      );
    }

    return this.trainiereTexte(
      daten,
      tokenizer,
      optionen
    );
  }
}

module.exports = {
  NeuronalesNetz
};
