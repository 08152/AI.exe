
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const STOPWOERTER = new Set([
  "der", "die", "das", "den", "dem", "des",
  "ein", "eine", "einer", "eines", "einem",
  "und", "oder", "aber", "ist", "sind", "war",
  "ich", "du", "er", "sie", "es", "wir", "ihr",
  "was", "wie", "wer", "wo", "wann", "warum",
  "mit", "von", "für", "auf", "in", "im", "am",
  "an", "zu", "zum", "zur", "auch", "nicht",
  "kein", "keine", "bitte", "noch", "schon",
  "sehr", "hat", "haben", "kann", "können"
]);

const VERBOTENE_AUSGABETOKENS = new Set([
  "<PAD>",
  "<BOS>",
  "<UNK>",
  "<SEP>",
  "<benutzer>",
  "<ki>"
]);

class NeuronalesNetz {
  constructor(tokenizer = null, optionen = {}) {
    if (
      tokenizer &&
      typeof tokenizer.zerlege !== "function"
    ) {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.tokenizer = tokenizer;

    this.maxVokabular = optionen.maxVokabular || 256;
    this.embeddingGroesse = optionen.embeddingGroesse || 8;

    // Das neuronale Netz besitzt jetzt 120 verborgene Neuronen.
    this.versteckteNeuronen = 120;

    this.kontextLaenge = optionen.kontextLaenge || 16;

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
    this.letzterFehler = null;

    // Die Trainingsbeispiele dienen auch zur Antwortplanung.
    this.trainingsPaare = [];
    this.planungsTreffer = 0;
  }

  // --------------------------------------------------
  // 1. TRAININGSDATEN EINLESEN
  // --------------------------------------------------

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

    const schluessel = {};

    for (const key of Object.keys(daten)) {
      schluessel[key.toLowerCase()] = key;
    }

    const frageFelder = [
      "frage", "question", "prompt", "input"
    ];

    const antwortFelder = [
      "antwort", "answer", "response",
      "completion", "output"
    ];

    const frageFeld = frageFelder.find(
      key => schluessel[key]
    );

    const antwortFeld = antwortFelder.find(
      key => schluessel[key]
    );

    if (
      frageFeld &&
      antwortFeld &&
      typeof daten[schluessel[frageFeld]] === "string" &&
      typeof daten[schluessel[antwortFeld]] === "string"
    ) {
      const frage = this.normalisiereText(
        daten[schluessel[frageFeld]]
      );

      const antwort = this.normalisiereText(
        daten[schluessel[antwortFeld]]
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

    const datei = path.join(
      __dirname,
      "modelle",
      "tokenizer.json"
    );

    this.tokenizer = fs.existsSync(datei)
      ? Tokenizer.laden(datei)
      : new Tokenizer();

    return this.tokenizer;
  }

  // --------------------------------------------------
  // 2. GEWICHTE INITIALISIEREN
  // --------------------------------------------------

  initialisiereGewichte() {
    const vokabularGroesse = this.vokabular.length;

    const eingabeGroesse =
      this.kontextLaenge * this.embeddingGroesse;

    const zufallsGewicht = grenze =>
      (Math.random() * 2 - 1) * grenze;

    this.embeddings = Array.from(
      { length: vokabularGroesse },
      () => Array.from(
        { length: this.embeddingGroesse },
        () => zufallsGewicht(0.1)
      )
    );

    const grenze1 = Math.sqrt(
      2 / (eingabeGroesse + this.versteckteNeuronen)
    );

    const grenze2 = Math.sqrt(
      2 / (this.versteckteNeuronen + vokabularGroesse)
    );

    // Eingabeschicht -> 120 verborgene Neuronen.
    this.gewichte1 = Array.from(
      { length: eingabeGroesse },
      () => Array.from(
        { length: this.versteckteNeuronen },
        () => zufallsGewicht(grenze1)
      )
    );

    this.bias1 = Array(
      this.versteckteNeuronen
    ).fill(0);

    // 120 verborgene Neuronen -> mögliche nächste Tokens.
    this.gewichte2 = Array.from(
      { length: this.versteckteNeuronen },
      () => Array.from(
        { length: vokabularGroesse },
        () => zufallsGewicht(grenze2)
      )
    );

    this.bias2 = Array(vokabularGroesse).fill(0);
  }

  begrenze(wert, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, wert));
  }

  // --------------------------------------------------
  // 3. VORWÄRTSBERECHNUNG
  // --------------------------------------------------

  vorwaerts(kontext) {
    const eingabe = [];

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
        eingabe.push(this.embeddings[id][d]);
      }
    }

    const versteckt = Array(
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

    const logits = Array(
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

    const maximum = Math.max(...logits);

    const exponenten = logits.map(wert =>
      Math.exp(Math.max(-60, wert - maximum))
    );

    const gesamt = exponenten.reduce(
      (summe, wert) => summe + wert,
      0
    );

    const wahrscheinlichkeiten = exponenten.map(
      wert => wert / (gesamt || 1)
    );

    return {
      eingabe,
      versteckt,
      logits,
      wahrscheinlichkeiten
    };
  }

  // --------------------------------------------------
  // 4. RÜCKWÄRTSBERECHNUNG UND LERNEN
  // --------------------------------------------------

  trainiereBeispiel(kontext, ziel, lernrate) {
    const ergebnis = this.vorwaerts(kontext);

    const gradAusgabe =
      ergebnis.wahrscheinlichkeiten.slice();

    gradAusgabe[ziel] -= 1;

    const gradVersteckt = Array(
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
        summe += this.gewichte2[h][v] * gradAusgabe[v];
      }

      gradVersteckt[h] = summe;
    }

    const gradVorAktivierung = gradVersteckt.map(
      (wert, h) =>
        wert * (1 - ergebnis.versteckt[h] ** 2)
    );

    const gradEingabe = Array(
      ergebnis.eingabe.length
    ).fill(0);

    // Eingabegradient berechnen, bevor Gewichte geändert werden.
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
          lernrate * ergebnis.versteckt[h] * fehler;
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
          lernrate * ergebnis.eingabe[i] * fehler;
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

  // --------------------------------------------------
  // 5. TRAINIEREN
  // --------------------------------------------------

  trainiereTexte(daten, tokenizer = null, optionen = {}) {
    if (
      tokenizer &&
      typeof tokenizer.zerlege !== "function"
    ) {
      optionen = tokenizer;
      tokenizer = null;
    }

    this.trainingsPaare = [];

    const texte = this.extrahiereTexte(daten);

    if (texte.length === 0) {
      throw new Error(
        "Keine gültigen Trainings-Texte gefunden."
      );
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
      throw new Error(
        "Das Vokabular ist zu klein."
      );
    }

    this.unkId = this.vokabular.indexOf("<UNK>");
    this.bosId = this.vokabular.indexOf("<BOS>");
    this.eosId = this.vokabular.indexOf("<EOS>");
    this.padId = this.vokabular.indexOf("<PAD>");

    if (this.unkId < 0) this.unkId = 1;
    if (this.bosId < 0) this.bosId = 2;
    if (this.eosId < 0) this.eosId = 3;
    if (this.padId < 0) this.padId = 0;

    this.bereit = false;
    this.initialisiereGewichte();

    const sequenzen = [];

    for (const element of texte) {
      const tokens = this.tokenizer.zerlege(
        element.text
      );

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
    let nummerGlobal = 0;

    for (const sequenz of sequenzen) {
      let kontext = Array(
        this.kontextLaenge
      ).fill(this.bosId);

      for (const ziel of sequenz) {
        if (nummerGlobal % schritt === 0) {
          beispiele.push({
            kontext: kontext.slice(),
            ziel
          });
        }

        kontext = kontext.slice(1).concat(ziel);
        nummerGlobal++;
      }
    }

    if (beispiele.length === 0) {
      throw new Error(
        "Es konnten keine Trainingsbeispiele erstellt werden."
      );
    }

    const epochen = Math.max(
      1,
      Math.min(
        20,
        Math.floor(optionen.epochen || 4)
      )
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

    try {
      this.tokenizer.speichern(
        path.join(__dirname, "modelle", "tokenizer.json")
      );
    } catch (fehler) {
      console.warn(
        "Tokenizer konnte nicht gespeichert werden:",
        fehler.message
      );
    }

    return this.status();
  }

  // --------------------------------------------------
  // 6. ANTWORTPLANUNG
  // --------------------------------------------------

  wichtigeWoerter(text) {
    return new Set(
      this.tokenizer.zerlege(text).filter(token => {
        return (
          /[\p{L}\p{N}]/u.test(token) &&
          token.length > 2 &&
          !STOPWOERTER.has(token) &&
          !token.startsWith("<")
        );
      })
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

    const normalisierteFrage = this.normalisiereText(frage);

    for (const paar of this.trainingsPaare) {
      let punktzahl = this.aehnlichkeit(
        normalisierteFrage,
        paar.frage
      );

      if (normalisierteFrage === paar.frage) {
        punktzahl = 1;
      }

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
      ziel: "Die Frage sinnvoll und verständlich beantworten.",
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

  // --------------------------------------------------
  // 7. NÄCHSTES TOKEN AUSWÄHLEN
  // --------------------------------------------------

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
        !VERBOTENE_AUSGABETOKENS.has(element.token)
      )
      .sort((a, b) => b.wert - a.wert)
      .slice(0, Math.max(1, topK));

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
    );

    let zufall = Math.random() * gesamt;

    for (let i = 0; i < kandidaten.length; i++) {
      zufall -= gewichte[i];

      if (zufall <= 0) {
        return kandidaten[i].id;
      }
    }

    return kandidaten[0].id;
  }

  // --------------------------------------------------
  // 8. EINEN VOLLSTÄNDIGEN KANDIDATEN ERZEUGEN
  // --------------------------------------------------

  generiereKandidaten(prompt, plan, optionen = {}) {
    const anzahl = Math.max(
      1,
      Math.min(6, optionen.anzahlKandidaten || 4)
    );

    const maxTokens = Math.max(
      1,
      Math.min(80, optionen.maxTokens || 35)
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

    const tokenStrings = this.tokenizer.zerlege(
      eingabetext
    );

    let startKontext = Array(
      this.kontextLaenge
    ).fill(this.bosId);

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
          !VERBOTENE_AUSGABETOKENS.has(token) &&
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

  // --------------------------------------------------
  // 9. DIE GANZE ANTWORT BEWERTEN
  // --------------------------------------------------

  bew ertePlatzhalter() {
    return 0;
  }

  bewerteAntwort(text, plan) {
    const tokens = this.tokenizer.zerlege(text);
    const woerter = tokens.filter(token =>
      /[\p{L}\p{N}]/u.test(token) &&
      !token.startsWith("<")
    );

    if (woerter.length === 0) {
      return -100;
    }

    const einzigartige = new Set(woerter);
    const vielfalt = einzigartige.size / woerter.length;

    // Mehr Abwechslung ist meist besser.
    let score = vielfalt * 2.0;

    // Extrem kurze oder extrem lange Antworten abwerten.
    if (woerter.length < 3) {
      score -= 2;
    } else if (woerter.length >= 5 && woerter.length <= 24) {
      score += 1;
    } else if (woerter.length > 35) {
      score -= 1.5;
    }

    // Wiederholte Wörter und Wortpaare bestrafen.
    const zaehler = new Map();

    for (const wort of woerter) {
      zaehler.set(wort, (zaehler.get(wort) || 0) + 1);
    }

    for (const anzahl of zaehler.values()) {
      if (anzahl > 1) {
        score -= (anzahl - 1) * 0.7;
      }
    }

    const bigrams = new Set();
    let wiederholtePaare = 0;

    for (let i = 1; i < woerter.length; i++) {
      const paar = `${woerter[i - 1]}|${woerter[i]}`;

      if (bigrams.has(paar)) {
        wiederholtePaare++;
      }

      bigrams.add(paar);
    }

    score -= wiederholtePaare * 1.5;

    // Relevanz zur geplanten Antwort erhöhen.
    const kandidatenWoerter = this.wichtigeWoerter(text);
    const planWoerter = new Set(plan.kernbegriffe);

    let gemeinsam = 0;

    for (const wort of kandidatenWoerter) {
      if (planWoerter.has(wort)) {
        gemeinsam++;
      }
    }

    score += Math.min(2, gemeinsam * 0.35);

    // Bei passendem Trainingsbeispiel den Inhalt vergleichen.
    if (plan.beispielAntwort) {
      score +=
        2.5 * this.aehnlichkeit(
          text,
          plan.beispielAntwort
        );
    }

    // Markierungen dürfen niemals in der sichtbaren Antwort sein.
    if (text.includes("<benutzer>") || text.includes("<ki>")) {
      score -= 10;
    }

    return score;
  }

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

  // --------------------------------------------------
  // 10. ANTWORT PLANEN, VOLLSTÄNDIG GENERIEREN,
  //     KANDIDATEN VERGLEICHEN UND BESTE ANTWORT WÄHLEN
  // --------------------------------------------------

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

    // Schritt A: Thema und relevante Trainingsbeispiele bestimmen.
    const plan = this.planeAntwort(prompt);

    this.planungsTreffer = plan.relevanz;

    // Exakte bekannte Frage: Die gespeicherte Antwort ist
    // bereits ein vollständiger, gelernter Antwortentwurf.
    if (
      plan.beispielAntwort &&
      plan.relevanz >= 0.99
    ) {
      return plan.beispielAntwort;
    }

    // Schritt B: Mehrere vollständige Antwortkandidaten erzeugen.
    const kandidaten = this.generiereKandidaten(
      prompt,
      plan,
      optionen
    );

    // Schritt C: Ganze Antworten bewerten, nicht nur das nächste Wort.
    kandidaten.sort((a, b) => b.bewertung - a.bewertung);

    const beste = kandidaten[0];

    // Schritt D: Wenn der Generator nur Wiederholungen erzeugt,
    // ein sehr passendes Trainingsbeispiel als Fallback nutzen.
    if (
      plan.beispielAntwort &&
      plan.relevanz >= 0.55 &&
      (!beste || beste.bewertung < 1.0)
    ) {
      return plan.beispielAntwort;
    }

    return beste && beste.text
      ? beste.text
      : "Ich konnte noch keine vollständige Antwort bilden.";
  }

  generiere(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  // --------------------------------------------------
  // 11. STATUS UND TRAININGSORDNER
  // --------------------------------------------------

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

  lerneOrdner(ordner, tokenizer = null, optionen = {}) {
    if (!fs.existsSync(ordner)) {
      throw new Error(
        `Trainingsordner nicht gefunden: ${ordner}`
      );
    }

    const dateien = fs.readdirSync(ordner, {
      withFileTypes: true
    });

    const daten = [];

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

        console.log("Trainingsdaten geladen:", datei.name);
      } catch (fehler) {
        console.error(
          `Datei ${datei.name} übersprungen:`,
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
