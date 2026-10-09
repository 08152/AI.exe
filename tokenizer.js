
"use strict";

// ============================================================
// tokenizer.js
// Eigener Tokenizer für MeineEigeneKI
// Keine externen Bibliotheken
//
// Funktionen:
// - Text normalisieren
// - Text in Tokens zerlegen
// - Tokens in Zahlen kodieren
// - Zahlen in Tokens zurückverwandeln
// - Wortschatz aus eigenen Textdaten aufbauen
// - Tokenizer speichern und laden
// ============================================================

const fs = require("node:fs");
const path = require("node:path");

class Tokenizer {
  constructor() {
    this.tokenZuId = new Map();
    this.idZuToken = [];

    this.spezialTokens = [
      "<PAD>",
      "<UNK>",
      "<BOS>",
      "<EOS>",
      "<SEP>"
    ];

    this.initialisiereSpezialTokens();
  }

  // ----------------------------------------------------------
  // Spezialtokens reservieren
  // ----------------------------------------------------------

  initialisiereSpezialTokens() {
    for (const token of this.spezialTokens) {
      this.fuegeTokenHinzu(token);
    }
  }

  // ----------------------------------------------------------
  // Text normalisieren
  // ----------------------------------------------------------

  normalisiere(text) {
    if (typeof text !== "string") {
      throw new TypeError("Der Text muss eine Zeichenkette sein.");
    }

    return text
      .normalize("NFC")
      .replace(/\r\n?/g, "\n")
      .trim();
  }

  // ----------------------------------------------------------
  // Text zerlegen
  //
  // Wörter, Zahlen und Satzzeichen bleiben getrennt.
  // Beispiel:
  // "Hallo, Welt!" -> ["Hallo", ",", "Welt", "!"]
  // ----------------------------------------------------------

  zerlege(text) {
    text = this.normalisiere(text);

    if (text.length === 0) {
      return [];
    }

    const treffer = text.match(
      /[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*|[\p{N}]+(?:[.,][\p{N}]+)*|[^\s\p{L}\p{M}\p{N}]/gu
    );

    return treffer || [];
  }

  // ----------------------------------------------------------
  // Token zum Wortschatz hinzufügen
  // ----------------------------------------------------------

  fuegeTokenHinzu(token) {
    if (this.tokenZuId.has(token)) {
      return this.tokenZuId.get(token);
    }

    const id = this.idZuToken.length;

    this.tokenZuId.set(token, id);
    this.idZuToken.push(token);

    return id;
  }

  // ----------------------------------------------------------
  // Wortschatz aus Texten lernen
  // ----------------------------------------------------------

  lerneTexte(textsammlung) {
    if (!Array.isArray(textsammlung)) {
      throw new TypeError(
        "Die Textsammlung muss ein Array sein."
      );
    }

    let neueTokens = 0;

    for (const text of textsammlung) {
      for (const token of this.zerlege(text)) {
        if (!this.tokenZuId.has(token)) {
          this.fuegeTokenHinzu(token);
          neueTokens++;
        }
      }
    }

    return {
      texte: textsammlung.length,
      neueTokens,
      wortschatzGroesse: this.idZuToken.length
    };
  }

  // ----------------------------------------------------------
  // Text in Token-IDs umwandeln
  //
  // Unbekannte Tokens werden zu <UNK>.
  // ----------------------------------------------------------

  kodieren(text, optionen = {}) {
    const {
      startToken = false,
      endeToken = false
    } = optionen;

    const tokens = this.zerlege(text);
    const ids = [];

    if (startToken) {
      ids.push(this.tokenZuId.get("<BOS>"));
    }

    for (const token of tokens) {
      const id = this.tokenZuId.get(token);

      ids.push(
        id === undefined
          ? this.tokenZuId.get("<UNK>")
          : id
      );
    }

    if (endeToken) {
      ids.push(this.tokenZuId.get("<EOS>"));
    }

    return ids;
  }

  // ----------------------------------------------------------
  // Token-IDs wieder in Text umwandeln
  // ----------------------------------------------------------

  dekodieren(ids, optionen = {}) {
    if (!Array.isArray(ids)) {
      throw new TypeError("Die Token-IDs müssen ein Array sein.");
    }

    const {
      spezialTokensAnzeigen = false
    } = optionen;

    const tokens = [];

    for (const id of ids) {
      if (
        !Number.isInteger(id) ||
        id < 0 ||
        id >= this.idZuToken.length
      ) {
        tokens.push("<UNK>");
        continue;
      }

      const token = this.idZuToken[id];

      if (
        !spezialTokensAnzeigen &&
        this.spezialTokens.includes(token)
      ) {
        continue;
      }

      tokens.push(token);
    }

    return this.setzeTokensZusammen(tokens);
  }

  // ----------------------------------------------------------
  // Tokens zu lesbarem Text zusammensetzen
  // ----------------------------------------------------------

  setzeTokensZusammen(tokens) {
    let text = "";

    const ohneLeerzeichenDavor = new Set([
      ".", ",", "!", "?", ";", ":", "%",
      ")", "]", "}", "…"
    ]);

    const ohneLeerzeichenDanach = new Set([
      "(", "[", "{"
    ]);

    for (const token of tokens) {
      if (text.length === 0) {
        text = token;
        continue;
      }

      if (ohneLeerzeichenDavor.has(token)) {
        text += token;
      } else if (ohneLeerzeichenDanach.has(text.slice(-1))) {
        text += token;
      } else {
        text += " " + token;
      }
    }

    return text;
  }

  // ----------------------------------------------------------
  // Token-ID abfragen
  // ----------------------------------------------------------

  idFuer(token) {
    const id = this.tokenZuId.get(token);

    return id === undefined
      ? this.tokenZuId.get("<UNK>")
      : id;
  }

  // ----------------------------------------------------------
  // Token zu einer ID abfragen
  // ----------------------------------------------------------

  tokenFuer(id) {
    if (
      !Number.isInteger(id) ||
      id < 0 ||
      id >= this.idZuToken.length
    ) {
      return "<UNK>";
    }

    return this.idZuToken[id];
  }

  // ----------------------------------------------------------
  // Wortschatzinformationen
  // ----------------------------------------------------------

  status() {
    return {
      wortschatzGroesse: this.idZuToken.length,
      spezialTokens: this.spezialTokens.slice()
    };
  }

  // ----------------------------------------------------------
  // Wortschatz als JSON speichern
  // ----------------------------------------------------------

  speichern(dateipfad) {
    const ziel = path.resolve(dateipfad);

    const daten = {
      version: 1,
      spezialTokens: this.spezialTokens,
      tokens: this.idZuToken
    };

    fs.mkdirSync(path.dirname(ziel), {
      recursive: true
    });

    fs.writeFileSync(
      ziel,
      JSON.stringify(daten, null, 2),
      "utf8"
    );

    return ziel;
  }

  // ----------------------------------------------------------
  // Wortschatz aus JSON laden
  // ----------------------------------------------------------

  static laden(dateipfad) {
    const quelle = path.resolve(dateipfad);

    const daten = JSON.parse(
      fs.readFileSync(quelle, "utf8")
    );

    if (
      !daten ||
      daten.version !== 1 ||
      !Array.isArray(daten.tokens) ||
      !Array.isArray(daten.spezialTokens)
    ) {
      throw new Error("Ungültige Tokenizer-Datei.");
    }

    const tokenizer = new Tokenizer();

    tokenizer.tokenZuId.clear();
    tokenizer.idZuToken = [];
    tokenizer.spezialTokens = daten.spezialTokens.slice();

    for (const token of daten.tokens) {
      if (
        typeof token !== "string" ||
        tokenizer.tokenZuId.has(token)
      ) {
        throw new Error(
          "Ungültiger oder doppelter Token in der Datei."
        );
      }

      tokenizer.fuegeTokenHinzu(token);
    }

    for (const token of tokenizer.spezialTokens) {
      if (!tokenizer.tokenZuId.has(token)) {
        throw new Error(
          "Spezialtoken fehlt im Wortschatz: " + token
        );
      }
    }

    return tokenizer;
  }
}

module.exports = { Tokenizer };
