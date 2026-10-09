
"use strict";

const fs = require("node:fs");
const path = require("node:path");

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
    this.versteckteNeuronen = optionen.versteckteNeuronen || 16;
    this.kontextLaenge = optionen.kontextLaenge || 12;

    this.embeddings = null;
    this.gewichte1 = null;
    this.bias1 = null;
    this.gewichte2 = null;
    this.bias2 = null;

    this.vokabular = [];
    this.tokenZuModellId = new Map();

    this.bereit = false;
    this.konversationsModus = false;
    this.trainingsBeispiele = 0;
    this.trainierteEpochen = 0;
    this.letzterFehler = null;
  }

  extrahiereTexte(daten, ergebnis = []) {
    if (typeof daten === "string") {
      if (daten.trim()) {
        ergebnis.push({
          text: daten.trim(),
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
      "antwort", "answer", "response", "completion", "output"
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
      const frage = daten[schluessel[frageFeld]].trim();
      const antwort = daten[schluessel[antwortFeld]].trim();

      if (frage && antwort) {
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
    if (this.tokenizer) return this.tokenizer;

    const { Tokenizer } = require("./tokenizer.js");
    const datei = path.join(
      __dirname,
      "modelle",
      "tokenizer.json"
    );

    if (fs.existsSync(datei)) {
      this.tokenizer = Tokenizer.laden(datei);
    } else {
      this.tokenizer = new Tokenizer();
    }

    return this.tokenizer;
  }

  idFuer(token) {
    const id = this.tokenizer.tokenZuId.get(token);

    if (
      Number.isInteger(id) &&
      id >= 0 &&
      id < this.vokabular.length
    ) {
      return id;
    }

    return this.unkId;
  }

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

    this.gewichte1 = Array.from(
      { length: eingabeGroesse },
      () => Array.from(
        { length: this.versteckteNeuronen },
        () => zufallsGewicht(grenze1)
      )
    );

    this.bias1 = Array(this.versteckteNeuronen).fill(0);

    this.gewichte2 = Array.from(
      { length: this.versteckteNeuronen },
      () => Array.from(
        { length: vokabularGroesse },
        () => zufallsGewicht(grenze2)
      )
    );

    this.bias2 = Array(vokabularGroesse).fill(0);
  }

  vorwaerts(kontext) {
    const eingabe = [];

    for (let position = 0; position < this.kontextLaenge; position++) {
      let id = kontext[position];

      if (
        !Number.isInteger(id) ||
        id < 0 ||
        id >= this.vokabular.length
      ) {
        id = this.unkId;
      }

      for (let d = 0; d < this.embeddingGroesse; d++) {
        eingabe.push(this.embeddings[id][d]);
      }
    }

    const versteckt = Array(this.versteckteNeuronen).fill(0);

    for (let h = 0; h < this.versteckteNeuronen; h++) {
      let summe = this.bias1[h];

      for (let i = 0; i < eingabe.length; i++) {
        summe += eingabe[i] * this.gewichte1[i][h];
      }

      versteckt[h] = Math.tanh(summe);
    }

    const logits = Array(this.vokabular.length).fill(0);

    for (let v = 0; v < this.vokabular.length; v++) {
      let summe = this.bias2[v];

      for (let h = 0; h < this.versteckteNeuronen; h++) {
        summe += versteckt[h] * this.gewichte2[h][v];
      }

      logits[v] = summe;
    }

    const maximum = Math.max(...logits);
    const exponenten = logits.map(
      wert => Math.exp(Math.max(-60, wert - maximum))
    );

    const summe = exponenten.reduce(
      (gesamt, wert) => gesamt + wert,
      0
    );

    const wahrscheinlichkeiten = exponenten.map(
      wert => wert / (summe || 1)
    );

    return {
      eingabe,
      versteckt,
      logits,
      wahrscheinlichkeiten
    };
  }

  trainiereBeispiel(kontext, ziel, lernrate) {
    const vorhersage = this.vorwaerts(kontext);
    const probs = vorhersage.wahrscheinlichkeiten;

    const gradAusgabe = probs.slice();
    gradAusgabe[ziel] -= 1;

    // Fehler rückwärts durch die Ausgabeschicht leiten.
    const gradVersteckt = Array(
      this.versteckteNeuronen
    ).fill(0);

    for (let h = 0; h < this.versteckteNeuronen; h++) {
      let summe = 0;

      for (let v = 0; v < this.vokabular.length; v++) {
        summe += this.gewichte2[h][v] * gradAusgabe[v];
      }

      gradVersteckt[h] = summe;
    }

    const gradVorAktivierung = gradVersteckt.map(
      (wert, h) =>
        wert * (1 - vorhersage.versteckt[h] ** 2)
    );

    const gradEingabe = Array(
      vorhersage.eingabe.length
    ).fill(0);

    for (let i = 0; i < vorhersage.eingabe.length; i++) {
      let summe = 0;

      for (let h = 0; h < this.versteckteNeuronen; h++) {
        summe +=
          this.gewichte1[i][h] * gradVorAktivierung[h];
      }

      gradEingabe[i] = Math.max(
        -5,
        Math.min(5, summe)
      );
    }

    // Ausgabeschicht anpassen.
    for (let v = 0; v < this.vokabular.length; v++) {
      const fehler = Math.max(
        -5,
        Math.min(5, gradAusgabe[v])
      );

      this.bias2[v] -= lernrate * fehler;

      for (let h = 0; h < this.versteckteNeuronen; h++) {
        this.gewichte2[h][v] -=
          lernrate * vorhersage.versteckt[h] * fehler;
      }
    }

    // Verborgene Schicht anpassen.
    for (let h = 0; h < this.versteckteNeuronen; h++) {
      const fehler = gradVorAktivierung[h];

      this.bias1[h] -= lernrate * fehler;

      for (let i = 0; i < vorhersage.eingabe.length; i++) {
        this.gewichte1[i][h] -=
          lernrate * vorhersage.eingabe[i] * fehler;
      }
    }

    // Auch die Wort-Embeddings lernen.
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

      for (let d = 0; d < this.embeddingGroesse; d++) {
        const index =
          position * this.embeddingGroesse + d;

        this.embeddings[id][d] -=
          lernrate * gradEingabe[index];
      }
    }
  }

  trainiereTexte(daten, tokenizer = null, optionen = {}) {
    if (
      tokenizer &&
      typeof tokenizer.zerlege !== "function"
    ) {
      optionen = tokenizer;
      tokenizer = null;
    }

    const texte = this.extrahiereTexte(daten);

    if (texte.length === 0) {
      throw new Error("Keine Trainings-Texte gefunden.");
    }

    this.tokenizer = tokenizer || this.ladeTokenizer();

    // Vokabular zuerst erweitern, dann das neuronale Netz aufbauen.
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

    this.tokenZuModellId = new Map();

    for (let id = 0; id < this.vokabular.length; id++) {
      this.tokenZuModellId.set(this.vokabular[id], id);
    }

    this.unkId = this.tokenZuModellId.get("<UNK>") ?? 1;
    this.bosId = this.tokenZuModellId.get("<BOS>") ?? 2;
    this.eosId = this.tokenZuModellId.get("<EOS>") ?? 3;
    this.padId = this.tokenZuModellId.get("<PAD>") ?? 0;

    if (this.vokabular.length < 5) {
      throw new Error(
        "Das Vokabular ist zu klein. Füge mehr Trainings-Texte hinzu."
      );
    }

    this.bereit = false;
    this.letzterFehler = null;
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
      optionen.maxTrainingsBeispiele || 1000
    );

    const schritt = Math.max(
      1,
      Math.ceil(anzahlZiele / maxBeispiele)
    );

    const beispiele = [];
    let nummer = 0;

    for (const sequenz of sequenzen) {
      let kontext = Array(this.kontextLaenge).fill(this.bosId);

      for (const ziel of sequenz) {
        if (nummer % schritt === 0) {
          beispiele.push({
            kontext: kontext.slice(),
            ziel
          });
        }

        kontext = kontext.slice(1).concat(ziel);
        nummer++;
      }
    }

    if (beispiele.length === 0) {
      throw new Error("Es konnten keine Trainingsbeispiele erstellt werden.");
    }

    const epochen = Math.max(
      1,
      Math.min(20, Math.floor(optionen.epochen || 4))
    );

    const lernrate = Number.isFinite(optionen.lernrate)
      ? optionen.lernrate
      : 0.025;

    for (let epoche = 0; epoche < epochen; epoche++) {
      // Trainingsreihenfolge mischen.
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
        `Neuronales Sprachmodell: Epoche ${epoche + 1}/${epochen}`
      );
    }

    this.trainingsBeispiele = beispiele.length;
    this.trainierteEpochen = epochen;
    this.bereit = true;

    // Das erweiterte Vokabular lokal ablegen.
    if (typeof this.tokenizer.speichern === "function") {
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
    }

    return this.status();
  }

  waehleNaechstesToken(kontext, temperatur = 0.85, topK = 12) {
    const ergebnis = this.vorwaerts(kontext);
    const temp = Math.max(0.1, Math.min(2, temperatur));

    const kandidaten = ergebnis.logits
      .map((wert, id) => ({
        id,
        wert: wert / temp,
        token: this.vokabular[id]
      }))
      .filter(element =>
        element.token !== "<PAD>" &&
        element.token !== "<BOS>" &&
        element.token !== "<UNK>"
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
      (summe, gewicht) => summe + gewicht,
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

  formatiere(tokens) {
    let text = tokens.join(" ");

    text = text
      .replace(/\s+([.,!?;:%)\]}»])/g, "$1")
      .replace(/([([{«])\s+/g, "$1")
      .trim();

    if (text) {
      text = text.charAt(0).toLocaleUpperCase("de-DE") +
        text.slice(1);
    }

    return text;
  }

  antwortGenerieren(prompt, optionen = {}) {
    if (!this.bereit) {
      return "Mein neuronales Netz ist noch nicht trainiert. " +
        "Füge Trainingsdaten hinzu und trainiere das Netz.";
    }

    if (typeof prompt !== "string" || !prompt.trim()) {
      return "Bitte gib eine Nachricht ein.";
    }

    const eingabetext = this.konversationsModus
      ? `<benutzer> ${prompt.trim()} <ki>`
      : prompt.trim();

    const tokenStrings = this.tokenizer.zerlege(eingabetext);

    const ids = tokenStrings.map(token => {
      const id = this.tokenizer.tokenZuId.get(token);

      return Number.isInteger(id) &&
        id < this.vokabular.length
        ? id
        : this.unkId;
    });

    let kontext = Array(this.kontextLaenge).fill(this.bosId);

    kontext = kontext.concat(ids).slice(-this.kontextLaenge);

    const maxTokens = Math.max(
      1,
      Math.min(120, optionen.maxTokens || 45)
    );

    const temperatur = optionen.temperatur || 0.85;
    const topK = optionen.topK || 12;
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
        token !== "<PAD>" &&
        token !== "<BOS>" &&
        token !== "<EOS>" &&
        token !== "<UNK>"
      ) {
        erzeugteTokens.push(token);
      }

      kontext = kontext.slice(1).concat(naechsteId);
    }

    return this.formatiere(erzeugteTokens);
  }

  generiere(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  status() {
    return {
      bereit: this.bereit,
      modell: "Neuronales Sprachmodell",
      vokabularGroesse: this.vokabular.length,
      trainingsBeispiele: this.trainingsBeispiele,
      trainierteEpochen: this.trainierteEpochen,
      kontextLaenge: this.kontextLaenge,
      konversationsModus: this.konversationsModus,
      letzterFehler: this.letzterFehler
    };
  }

  lerneOrdner(ordner, tokenizer = null, optionen = {}) {
    if (!fs.existsSync(ordner)) {
      throw new Error(`Trainingsordner nicht gefunden: ${ordner}`);
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
        const dateipfad = path.join(ordner, datei.name);
        daten.push(
          JSON.parse(fs.readFileSync(dateipfad, "utf8"))
        );

        console.log("Sprachdaten geladen:", datei.name);
      } catch (fehler) {
        console.error(
          "Datei übersprungen:",
          datei.name,
          fehler.message
        );
      }
    }

    return this.trainiereTexte(daten, tokenizer, optionen);
  }
}

module.exports = {
  NeuronalesNetz
};
