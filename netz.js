"use strict";

const fs = require("node:fs");
const path = require("node:path");

const STOPWOERTER = new Set([
  "der", "die", "das", "den", "dem", "des", "ein", "eine",
  "einer", "eines", "einem", "und", "oder", "aber", "ist",
  "sind", "war", "waren", "ich", "du", "er", "sie", "es",
  "wir", "ihr", "was", "wie", "wer", "wo", "wann", "warum",
  "wieso", "mit", "von", "für", "auf", "in", "im", "am", "an",
  "zu", "zum", "zur", "auch", "nicht", "kein", "keine", "bitte",
  "noch", "schon", "sehr", "hat", "haben", "kann", "können",
  "sich", "mir", "mich", "mein", "meine", "dein", "deine"
]);

const SPEZIAL_AUSGABE_VERBOTEN = new Set([
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

    this.tokenizer = tokenizer;

    this.maxVokabular = Number.isInteger(optionen.maxVokabular)
      ? Math.max(16, optionen.maxVokabular)
      : 512;

    this.embeddingGroesse = Number.isInteger(optionen.embeddingGroesse)
      ? Math.max(2, optionen.embeddingGroesse)
      : 8;

    this.versteckteNeuronen = Number.isInteger(optionen.versteckteNeuronen)
      ? Math.max(8, optionen.versteckteNeuronen)
      : 120;

    this.kontextLaenge = Number.isInteger(optionen.kontextLaenge)
      ? Math.max(2, optionen.kontextLaenge)
      : 16;

    this.vokabular = [];
    this.embeddings = null;
    this.gewichte1 = null;
    this.bias1 = null;
    this.gewichte2 = null;
    this.bias2 = null;

    this.unkId = 1;
    this.bosId = 2;
    this.eosId = 3;
    this.padId = 0;

    this.bereit = false;
    this.konversationsModus = false;
    this.trainingsBeispiele = 0;
    this.trainierteEpochen = 0;
    this.trainingsPaare = [];
    this.letzteAntworten = [];
    this.letzterFehler = null;
  }

  // -----------------------------------------------
  // TEXT UND TRAININGSDATEN
  // -----------------------------------------------

  normalisiereText(text) {
    return text
      .normalize("NFC")
      .toLocaleLowerCase("de-DE")
      .trim();
  }

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

    const keys = Object.keys(daten);
    const nachKlein = Object.create(null);

    for (const key of keys) {
      nachKlein[key.toLowerCase()] = key;
    }

    const frageFelder = [
      "frage",
      "question",
      "prompt",
      "input"
    ];

    const antwortFelder = [
      "antwort",
      "answer",
      "response",
      "completion",
      "output"
    ];

    const frageFeld = frageFelder.find(
      key => nachKlein[key]
    );

    const antwortFeld = antwortFelder.find(
      key => nachKlein[key]
    );

    if (
      frageFeld &&
      antwortFeld &&
      typeof daten[nachKlein[frageFeld]] === "string" &&
      typeof daten[nachKlein[antwortFeld]] === "string"
    ) {
      const frage = this.normalisiereText(
        daten[nachKlein[frageFeld]]
      );

      const antwort = this.normalisiereText(
        daten[nachKlein[antwortFeld]]
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

  ladeTokenizer() {
    if (this.tokenizer) {
      return this.tokenizer;
    }

    const { Tokenizer } = require("./tokenizer.js");

    const dateipfad = path.join(
      __dirname,
      "modelle",
      "tokenizer.json"
    );

    this.tokenizer = fs.existsSync(dateipfad)
      ? Tokenizer.laden(dateipfad)
      : new Tokenizer();

    return this.tokenizer;
  }

  // -----------------------------------------------
  // GEWICHTE INITIALISIEREN
  // -----------------------------------------------

  initialisiereGewichte() {
    const vokabularGroesse = this.vokabular.length;

    const eingabeGroesse =
      this.kontextLaenge * this.embeddingGroesse;

    const zufall = grenze =>
      (Math.random() * 2 - 1) * grenze;

    this.embeddings = Array.from(
      { length: vokabularGroesse },
      () => Array.from(
        { length: this.embeddingGroesse },
        () => zufall(0.1)
      )
    );

    const grenze1 = Math.sqrt(
      2 / (eingabeGroesse + this.versteckteNeuronen)
    );

    const grenze2 = Math.sqrt(
      2 / (this.versteckteNeuronen + vokabularGroesse)
    );

    this.gewichte1 = Array.from(
      { length: eingabeGroesse },
      () => Array.from(
        { length: this.versteckteNeuronen },
        () => zufall(grenze1)
      )
    );

    this.bias1 = Array(
      this.versteckteNeuronen
    ).fill(0);

    this.gewichte2 = Array.from(
      { length: this.versteckteNeuronen },
      () => Array.from(
        { length: vokabularGroesse },
        () => zufall(grenze2)
      )
    );

    this.bias2 = Array(vokabularGroesse).fill(0);
  }

  begrenze(wert, minimum, maximum) {
    return Math.max(
      minimum,
      Math.min(maximum, wert)
    );
  }

  // -----------------------------------------------
  // VORWÄRTSBERECHNUNG
  // -----------------------------------------------

  vorwaerts(kontext) {
    const eingabe = new Array(
      this.kontextLaenge * this.embeddingGroesse
    );

    let positionIndex = 0;

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

      const embedding = this.embeddings[id];

      for (
        let d = 0;
        d < this.embeddingGroesse;
        d++
      ) {
        eingabe[positionIndex++] = embedding[d];
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
        summe += versteckt[h] * this.gewichte2[h][v];
      }

      logits[v] = summe;
    }

    let maximum = -Infinity;

    for (const wert of logits) {
      if (wert > maximum) maximum = wert;
    }

    const exponenten = logits.map(
      wert => Math.exp(Math.max(-60, wert - maximum))
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

  // -----------------------------------------------
  // RÜCKWÄRTSBERECHNUNG UND LERNEN
  // -----------------------------------------------

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
          this.gewichte2[h][v] * gradAusgabe[v];
      }

      gradVersteckt[h] = summe;
    }

    const gradVorAktivierung = gradVersteckt.map(
      (wert, h) =>
        wert * (
          1 -
          ergebnis.versteckt[h] *
          ergebnis.versteckt[h]
        )
    );

    const gradEingabe = new Array(
      ergebnis.eingabe.length
    ).fill(0);

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

      gradEingabe[i] = this.begrenze(summe, -5, 5);
    }

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
    // -----------------------------------------------
  // SPRACHMODELL TRAINIEREN
  // -----------------------------------------------

  trainiereTexte(daten, tokenizer = null, optionen = {}) {
    if (
      tokenizer &&
      typeof tokenizer.zerlege !== "function"
    ) {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.trainingsPaare = [];
    this.bereit = false;
    this.letzterFehler = null;

    const texte = this.extrahiereTexte(daten);

    if (texte.length === 0) {
      throw new Error("Keine Trainings-Texte gefunden.");
    }

    this.tokenizer = tokenizer || this.ladeTokenizer();

    this.tokenizer.lerneTexte(
      texte.map(element => element.text)
    );

    this.konversationsModus = texte.some(
      element => element.istKonversation
    );

    this.vokabular = this.tokenizer.idZuToken.slice(
      0,
      this.maxVokabular
    );

    if (this.vokabular.length < 5) {
      throw new Error("Das Vokabular ist zu klein.");
    }

    const findeId = (token, fallback) => {
      const id = this.vokabular.indexOf(token);
      return id >= 0 ? id : fallback;
    };

    this.padId = findeId("<PAD>", 0);
    this.unkId = findeId("<UNK>", 1);
    this.bosId = findeId("<BOS>", 2);
    this.eosId = findeId("<EOS>", 3);

    this.initialisiereGewichte();

    const sequenzen = [];

    for (const element of texte) {
      const tokens = this.tokenizer.zerlege(element.text);

      const ids = tokens.map(token => {
        const id = this.tokenizer.tokenZuId.get(token);

        return Number.isInteger(id) &&
          id < this.vokabular.length
          ? id
          : this.unkId;
      });

      if (ids.length > 0) {
        sequenzen.push([...ids, this.eosId]);
      }
    }

    const anzahlZiele = sequenzen.reduce(
      (summe, sequenz) => summe + sequenz.length,
      0
    );

    const maxBeispiele = Math.max(
      1,
      Math.floor(optionen.maxTrainingsBeispiele || 700)
    );

    const schritt = Math.max(
      1,
      Math.ceil(anzahlZiele / maxBeispiele)
    );

    const beispiele = [];
    let globalePosition = 0;

    for (const sequenz of sequenzen) {
      let kontext = Array(this.kontextLaenge).fill(this.bosId);

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
      throw new Error("Keine Trainingsbeispiele erstellt.");
    }

    const epochen = Math.max(
      1,
      Math.min(30, Math.floor(optionen.epochen || 4))
    );

    const lernrate = Number.isFinite(optionen.lernrate)
      ? optionen.lernrate
      : 0.015;

    for (let epoche = 0; epoche < epochen; epoche++) {
      for (let i = beispiele.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));

        [beispiele[i], beispiele[j]] =
          [beispiele[j], beispiele[i]];
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

    if (typeof this.tokenizer.speichern === "function") {
      const datei = path.join(
        __dirname,
        "modelle",
        "tokenizer.json"
      );

      try {
        this.tokenizer.speichern(datei);
      } catch (fehler) {
        console.warn(
          "Tokenizer konnte nicht gespeichert werden:",
          fehler.message
        );
      }
    }

    return this.status();
  }

  // -----------------------------------------------
  // ANTWORTPLANUNG UND RELEVANTE WÖRTER
  // -----------------------------------------------

  wichtigeWoerter(text) {
    return new Set(
      this.tokenizer.zerlege(text).filter(token =>
        /[\p{L}\p{N}]/u.test(token) &&
        token.length > 2 &&
        !STOPWOERTER.has(token) &&
        !token.startsWith("<")
      )
    );
  }

  aehnlichkeit(textA, textB) {
    const a = this.wichtigeWoerter(textA);
    const b = this.wichtigeWoerter(textB);

    if (a.size === 0 || b.size === 0) {
      return 0;
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

    const normalisiert = this.normalisiereText(frage);

    for (const paar of this.trainingsPaare) {
      const punktzahl = normalisiert === paar.frage
        ? 1
        : this.aehnlichkeit(normalisiert, paar.frage);

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

  planeAntwort(prompt) {
    const treffer = this.findePassendesBeispiel(prompt);
    const kernbegriffe = new Set(
      this.wichtigeWoerter(prompt)
    );

    if (treffer.paar && treffer.punktzahl >= 0.3) {
      for (
        const wort of this.wichtigeWoerter(
          treffer.paar.antwort
        )
      ) {
        kernbegriffe.add(wort);
      }
    }

    return {
      eingabe: prompt,
      ziel: "Eine zusammenhängende Antwort erzeugen.",
      kernbegriffe: [...kernbegriffe],
      beispielAntwort:
        treffer.paar && treffer.punktzahl >= 0.3
          ? treffer.paar.antwort
          : null,
      beispielFrage:
        treffer.paar && treffer.punktzahl >= 0.3
          ? treffer.paar.frage
          : null,
      relevanz: treffer.punktzahl
    };
  }

  // -----------------------------------------------
  // NÄCHSTES TOKEN AUSWÄHLEN
  // -----------------------------------------------

  waehleNaechstesToken(
    kontext,
    temperatur = 0.65,
    topK = 5
  ) {
    const ergebnis = this.vorwaerts(kontext);

    const temp = Math.max(
      0.1,
      Math.min(2, temperatur)
    );

    const kandidaten = ergebnis.logits
      .map((wert, id) => ({
        id,
        wert: wert / temp,
        token: this.vokabular[id]
      }))
      .filter(element =>
        !SPEZIAL_AUSGABE_VERBOTEN.has(element.token)
      )
      .sort((a, b) => b.wert - a.wert)
      .slice(0, Math.max(1, Math.floor(topK)));

    if (kandidaten.length === 0) {
      return this.eosId;
    }

    const maximum = kandidaten[0].wert;

    const gewichte = kandidaten.map(element =>
      Math.exp(Math.max(-60, element.wert - maximum))
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

  // -----------------------------------------------
  // TEXTE AUS TOKEN-FOLGEN FORMATIEREN
  // -----------------------------------------------

  formatiere(tokens) {
    let text = tokens.join(" ");

    text = text
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

  // -----------------------------------------------
  // MEHRERE GANZE ANTWORTKANDIDATEN GENERIEREN
  // -----------------------------------------------

  generiereKandidaten(prompt, plan, optionen = {}) {
    const anzahl = Math.max(
      1,
      Math.min(
        6,
        Math.floor(optionen.anzahlKandidaten || 4)
      )
    );

    const maxTokens = Math.max(
      1,
      Math.min(
        80,
        Math.floor(optionen.maxTokens || 35)
      )
    );

    const temperatur = Number.isFinite(optionen.temperatur)
      ? optionen.temperatur
      : 0.65;

    const topK = Number.isFinite(optionen.topK)
      ? optionen.topK
      : 5;

    const eingabetext = this.konversationsModus
      ? `<benutzer> ${this.normalisiereText(prompt)} <ki>`
      : this.normalisiereText(prompt);

    const tokenStrings = this.tokenizer.zerlege(eingabetext);

    let startKontext = Array(this.kontextLaenge).fill(this.bosId);

    for (const token of tokenStrings) {
      const originalId = this.tokenizer.tokenZuId.get(token);

      const id =
        Number.isInteger(originalId) &&
        originalId < this.vokabular.length
          ? originalId
          : this.unkId;

      startKontext = startKontext.slice(1).concat(id);
    }

    const kandidaten = [];

    for (let versuch = 0; versuch < anzahl; versuch++) {
      let kontext = startKontext.slice();
      const erzeugteTokens = [];

      for (let i = 0; i < maxTokens; i++) {
        const naechsteId = this.waehleNaechstesToken(
          kontext,
          temperatur,
          topK
        );

        if (naechsteId === this.eosId) {
          break;
        }

        const token = this.vokabular[naechsteId];

        if (
          token &&
          !SPEZIAL_AUSGABE_VERBOTEN.has(token) &&
          token !== "<EOS>"
        ) {
          erzeugteTokens.push(token);
        }

        kontext = kontext.slice(1).concat(naechsteId);
      }

      const text = this.formatiere(erzeugteTokens);

      kandidaten.push({
        text,
        bewertung: this.bewerteAntwort(text, plan)
      });
    }

    return kandidaten;
  }
    // -----------------------------------------------
  // GANZE ANTWORTEN BEWERTEN
  // -----------------------------------------------

  bewerteAntwort(text, plan) {
    const tokens = this.tokenizer.zerlege(text);

    const woerter = tokens.filter(token =>
      /[\p{L}\p{N}]/u.test(token) &&
      !token.startsWith("<")
    );

    if (woerter.length === 0) {
      return -100;
    }

    const vielfalt =
      new Set(woerter).size / woerter.length;

    let score = vielfalt * 2;

    if (woerter.length < 3) {
      score -= 2;
    } else if (woerter.length >= 5 && woerter.length <= 24) {
      score += 1;
    } else if (woerter.length > 35) {
      score -= 1.5;
    }

    const zaehler = new Map();

    for (const wort of woerter) {
      zaehler.set(wort, (zaehler.get(wort) || 0) + 1);
    }

    for (const anzahl of zaehler.values()) {
      if (anzahl > 1) {
        score -= (anzahl - 1) * 0.7;
      }
    }

    const paare = new Set();

    for (let i = 1; i < woerter.length; i++) {
      const paar = `${woerter[i - 1]}|${woerter[i]}`;

      if (paare.has(paar)) {
        score -= 1.5;
      }

      paare.add(paar);
    }

    const antwortWoerter = this.wichtigeWoerter(text);
    const planWoerter = new Set(plan.kernbegriffe);

    let gemeinsam = 0;

    for (const wort of antwortWoerter) {
      if (planWoerter.has(wort)) gemeinsam++;
    }

    score += Math.min(2, gemeinsam * 0.35);

    if (plan.beispielAntwort) {
      score += 2.5 * this.aehnlichkeit(
        text,
        plan.beispielAntwort
      );
    }

    if (text.includes("<benutzer>") || text.includes("<ki>")) {
      score -= 10;
    }

    return score;
  }

  // -----------------------------------------------
  // ÄLTERE ANTWORTMETHODE
  // -----------------------------------------------

  antwortGenerierenAlt(prompt, optionen = {}) {
    if (!this.bereit) {
      return (
        "Mein neuronales Sprachmodell ist noch nicht trainiert. " +
        "Bitte überprüfe deine Trainingsdaten."
      );
    }

    if (typeof prompt !== "string" || !prompt.trim()) {
      return "Bitte gib eine Nachricht ein.";
    }

    const plan = this.planeAntwort(prompt);

    if (plan.beispielAntwort && plan.relevanz >= 0.999) {
      return plan.beispielAntwort;
    }

    const kandidaten = this.generiereKandidaten(
      prompt,
      plan,
      optionen
    );

    kandidaten.sort((a, b) => b.bewertung - a.bewertung);

    const beste = kandidaten[0];

    if (
      plan.beispielAntwort &&
      plan.relevanz >= 0.55 &&
      (!beste || beste.bewertung < 1)
    ) {
      return plan.beispielAntwort;
    }

    return beste && beste.text
      ? beste.text
      : "Ich konnte noch keine zusammenhängende Antwort erzeugen.";
  }

  // -----------------------------------------------
  // WAHRSCHEINLICHSTE ANTWORT BERECHNEN
  // -----------------------------------------------

  antwortGenerieren(prompt, optionen = {}) {
    if (!this.bereit) {
      return (
        "Mein neuronales Sprachmodell ist noch nicht trainiert. " +
        "Bitte überprüfe deine Trainingsdaten."
      );
    }

    if (typeof prompt !== "string" || !prompt.trim()) {
      return "Bitte gib eine Nachricht ein.";
    }

    const frage = this.normalisiereText(prompt);
    const kandidaten = [];

    // Jede gespeicherte Frage mit der Eingabe vergleichen.
    for (const paar of this.trainingsPaare) {
      const exakt = frage === this.normalisiereText(paar.frage);
      const aehnlichkeit = exakt
        ? 1
        : this.aehnlichkeit(frage, paar.frage);

      kandidaten.push({
        frage: paar.frage,
        antwort: paar.antwort,
        exakt,
        wahrscheinlichkeit: aehnlichkeit
      });
    }

    // Bei exakt gleicher Frage die häufigste Antwort nehmen.
    const exakteTreffer = kandidaten.filter(k => k.exakt);

    if (exakteTreffer.length > 0) {
      const haeufigkeiten = new Map();

      for (const kandidat of exakteTreffer) {
        const antwort = kandidat.antwort;

        haeufigkeiten.set(
          antwort,
          (haeufigkeiten.get(antwort) || 0) + 1
        );
      }

      let besteAntwort = exakteTreffer[0].antwort;
      let besteHaeufigkeit = 0;

      for (const [antwort, anzahl] of haeufigkeiten) {
        if (anzahl > besteHaeufigkeit) {
          besteAntwort = antwort;
          besteHaeufigkeit = anzahl;
        }
      }

      return besteAntwort;
    }

    // Bei ähnlichen Fragen den besten Treffer auswählen.
    kandidaten.sort(
      (a, b) => b.wahrscheinlichkeit - a.wahrscheinlichkeit
    );

    const besterTreffer = kandidaten[0];

    if (
      besterTreffer &&
      besterTreffer.wahrscheinlichkeit >= 0.3
    ) {
      return besterTreffer.antwort;
    }

    // Unbekannte Frage: jeweils das wahrscheinlichste Token wählen.
    const maxTokens = Math.max(
      1,
      Math.min(80, Math.floor(optionen.maxTokens || 35))
    );

    const eingabetext = this.konversationsModus
      ? `<benutzer> ${frage} <ki>`
      : frage;

    const tokenStrings = this.tokenizer.zerlege(eingabetext);

    let kontext = Array(this.kontextLaenge).fill(this.bosId);

    for (const token of tokenStrings) {
      const originalId = this.tokenizer.tokenZuId.get(token);

      const id =
        Number.isInteger(originalId) &&
        originalId >= 0 &&
        originalId < this.vokabular.length
          ? originalId
          : this.unkId;

      kontext = kontext.slice(1).concat(id);
    }

    const erzeugteTokens = [];

    for (let i = 0; i < maxTokens; i++) {
      const ergebnis = this.vorwaerts(kontext);

      let besteId = -1;
      let besteWahrscheinlichkeit = -Infinity;

      for (
        let id = 0;
        id < ergebnis.wahrscheinlichkeiten.length;
        id++
      ) {
        const token = this.vokabular[id];

        if (!token || SPEZIAL_AUSGABE_VERBOTEN.has(token)) {
          continue;
        }

        const wahrscheinlichkeit =
          ergebnis.wahrscheinlichkeiten[id];

        if (wahrscheinlichkeit > besteWahrscheinlichkeit) {
          besteWahrscheinlichkeit = wahrscheinlichkeit;
          besteId = id;
        }
      }

      if (besteId < 0 || besteId === this.eosId) {
        break;
      }

      const token = this.vokabular[besteId];

      if (token && token !== "<EOS>") {
        erzeugteTokens.push(token);
      }

      kontext = kontext.slice(1).concat(besteId);
    }

    const antwort = this.formatiere(erzeugteTokens);

    if (!antwort.trim()) {
      return (
        "Dazu kenne ich noch keine passende Antwort. " +
        "Ich muss mehr Trainingsbeispiele lernen."
      );
    }

    return antwort;
  }

  generiere(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  // -----------------------------------------------
  // STATUS
  // -----------------------------------------------

  status() {
    return {
      bereit: this.bereit,
      modell: "Neuronales Sprachmodell mit Antwortplanung",
      versteckteNeuronen: this.versteckteNeuronen,
      vokabularGroesse: this.vokabular.length,
      trainingsBeispiele: this.trainingsBeispiele,
      trainierteEpochen: this.trainierteEpochen,
      kontextLaenge: this.kontextLaenge,
      konversationsModus: this.konversationsModus,
      trainingsPaare: this.trainingsPaare.length,
      letzterFehler: this.letzterFehler
    };
  }

  // -----------------------------------------------
  // TRAININGSORDNER LADEN
  // -----------------------------------------------

  lerneOrdner(ordner, tokenizer = null, optionen = {}) {
    if (!fs.existsSync(ordner)) {
      throw new Error(
        `Trainingsordner nicht gefunden: ${ordner}`
      );
    }

    const daten = [];

    const dateien = fs.readdirSync(ordner, {
      withFileTypes: true
    });

    for (const datei of dateien) {
      if (
        !datei.isFile() ||
        !datei.name.toLowerCase().endsWith(".json") ||
        datei.name.toLowerCase() === "tokenizer.json"
      ) {
        continue;
      }

      try {
        daten.push(
          JSON.parse(
            fs.readFileSync(
              path.join(ordner, datei.name),
              "utf8"
            )
          )
        );
      } catch (fehler) {
        console.error(
          `Trainingsdatei ${datei.name} übersprungen:`,
          fehler.message
        );
      }
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
