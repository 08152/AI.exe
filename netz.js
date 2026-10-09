
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

    this.vokabular = [];
    this.unkId = 1;
    this.bosId = 2;
    this.eosId = 3;
    this.padId = 0;

    this.embeddings = null;
    this.gewichte1 = null;
    this.bias1 = null;
    this.gewichte2 = null;
    this.bias2 = null;

    this.bereit = false;
    this.konversationsModus = false;
    this.trainingsBeispiele = 0;
    this.trainierteEpochen = 0;
    this.letzterFehler = null;
  }

  // ------------------------------------------------
  // TEXT UND TRAININGSDATEN
  // ------------------------------------------------

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

  // ------------------------------------------------
  // NETZWERK INITIALISIEREN
  // ------------------------------------------------

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

    // Eingang -> verborgene Schicht
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

    // Verborgene Schicht -> Wortwahrscheinlichkeiten
    this.gewichte2 = Array.from(
      { length: this.versteckteNeuronen },
      () => Array.from(
        { length: vokabularGroesse },
        () => zufallsGewicht(grenze2)
      )
    );

    this.bias2 = Array(vokabularGroesse).fill(0);
  }

  // ------------------------------------------------
  // VORWÄRTSBERECHNUNG
  // ------------------------------------------------

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

    // Softmax: Aus Ausgabewerten werden Wahrscheinlichkeiten.
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

  // ------------------------------------------------
  // RÜCKWÄRTSBERECHNUNG: GEWICHTE LERNEN
  // ------------------------------------------------

  trainiereBeispiel(kontext, ziel, lernrate) {
    const ergebnis = this.vorwaerts(kontext);

    const gradAusgabe =
      ergebnis.wahrscheinlichkeiten.slice();

    gradAusgabe[ziel] -= 1;

    const gradVersteckt = Array(
      this.versteckteNeuronen
    ).fill(0);

    // Fehler zurück durch die Ausgabeschicht.
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

  begrenze(wert, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, wert));
  }

  // ------------------------------------------------
  // SPRACHMODELL TRAINIEREN
  // ------------------------------------------------

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
      throw new Error(
        "Keine passenden Text- oder Frage-Antwort-Daten gefunden."
      );
    }

    this.tokenizer = tokenizer || this.ladeTokenizer();

    // Das Vokabular vor der Initialisierung erweitern.
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
        "Das Vokabular ist zu klein. Füge mehr Trainingsdaten hinzu."
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
    this.letzterFehler = null;

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

    // Aus jedem Satz viele Kontext -> nächstes Token Beispiele bauen.
    const beispiele = [];
    const maxBeispiele = Math.max(
      1,
      Math.floor(optionen.maxTrainingsBeispiele || 1000)
    );

    const anzahlZiele = sequenzen.reduce(
      (summe, sequenz) => summe + sequenz.length,
      0
    );

    const schritt = Math.max(
      1,
      Math.ceil(anzahlZiele / maxBeispiele)
    );

    let positionGlobal = 0;

    for (const sequenz of sequenzen) {
      let kontext = Array(
        this.kontextLaenge
      ).fill(this.bosId);

      for (const ziel of sequenz) {
        if (positionGlobal % schritt === 0) {
          beispiele.push({
            kontext: kontext.slice(),
            ziel
          });
        }

        kontext = kontext.slice(1).concat(ziel);
        positionGlobal++;
      }
    }

    if (beispiele.length === 0) {
      throw new Error(
        "Es konnten keine Trainingsbeispiele erstellt werden."
      );
    }

    const epochen = Math.max(
      1,
      Math.min(30, Math.floor(optionen.epochen || 4))
    );

    const lernrate = Number.isFinite(optionen.lernrate)
      ? optionen.lernrate
      : 0.025;

    for (let epoche = 0; epoche < epochen; epoche++) {
      // Trainingsbeispiele mischen.
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
        `Sprachmodell: Epoche ${epoche + 1}/${epochen}, ` +
        `${beispiele.length} Beispiele`
      );
    }

    this.trainingsBeispiele = beispiele.length;
    this.trainierteEpochen = epochen;
    this.bereit = true;

    // Vokabular lokal speichern.
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

  // ------------------------------------------------
  // NÄCHSTES TOKEN AUSWÄHLEN
  // ------------------------------------------------

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

    const verboteneTokens = new Set([
      "<PAD>",
      "<BOS>",
      "<UNK>",
      "<SEP>",
      "<benutzer>",
      "<ki>"
    ]);

    const kandidaten = ergebnis.logits
      .map((wert, id) => ({
        id,
        wert: wert / temp,
        token: this.vokabular[id]
      }))
      .filter(element =>
        !verboteneTokens.has(element.token)
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

  // ------------------------------------------------
  // TEXT AUS TOKENS ERZEUGEN
  // ------------------------------------------------

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

  antwortGenerieren(prompt, optionen = {}) {
    if (!this.bereit) {
      return (
        "Mein neuronales Sprachmodell ist noch nicht trainiert. " +
        "Bitte überprüfe die Trainingsdaten."
      );
    }

    if (typeof prompt !== "string" || !prompt.trim()) {
      return "Bitte gib eine Nachricht ein.";
    }

    const text = this.normalisiereText(prompt);

    let tokenStrings = this.tokenizer.zerlege(text);

    // Bei Frage-Antwort-Daten das Gesprächsformat benutzen.
    if (this.konversationsModus) {
      tokenStrings = [
        "<benutzer>",
        ...tokenStrings,
        "<ki>"
      ];
    }

    let kontext = Array(
      this.kontextLaenge
    ).fill(this.bosId);

    for (const token of tokenStrings) {
      const idOriginal = this.tokenizer.tokenZuId.get(token);

      const id =
        Number.isInteger(idOriginal) &&
        idOriginal < this.vokabular.length
          ? idOriginal
          : this.unkId;

      kontext = kontext.slice(1).concat(id);
    }

    const maxTokens = Math.max(
      1,
      Math.min(120, Math.floor(optionen.maxTokens || 45))
    );

    const temperatur = Number.isFinite(optionen.temperatur)
      ? optionen.temperatur
      : 0.65;

    const topK = Number.isFinite(optionen.topK)
      ? optionen.topK
      : 5;

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
        ![
          "<PAD>",
          "<BOS>",
          "<UNK>",
          "<EOS>",
          "<SEP>",
          "<benutzer>",
          "<ki>"
        ].includes(token)
      ) {
        erzeugteTokens.push(token);
      }

      kontext = kontext.slice(1).concat(naechsteId);
    }

    return this.formatiere(erzeugteTokens) ||
      "Ich konnte noch keine passende Wortfolge erzeugen.";
  }

  generiere(prompt, optionen = {}) {
    return this.antwortGenerieren(prompt, optionen);
  }

  // ------------------------------------------------
  // STATUS UND TRAININGSORDNER
  // ------------------------------------------------

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
        const dateipfad = path.join(ordner, datei.name);

        daten.push(
          JSON.parse(fs.readFileSync(dateipfad, "utf8"))
        );

        console.log("Trainingsdaten geladen:", datei.name);
      } catch (fehler) {
        console.error(
          "Datei übersprungen:",
          datei.name,
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
