
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const SPEZIAL_TOKENS = [
  "<PAD>",
  "<UNK>",
  "<BOS>",
  "<EOS>",
  "<SEP>"
];

class Tokenizer {
  constructor() {
    this.tokenZuId = new Map();
    this.idZuToken = [];

    for (const token of SPEZIAL_TOKENS) {
      this._registriereToken(token);
    }
  }

  static get standardDateipfad() {
    return process.env.TOKENIZER_PATH ||
      path.join(__dirname, "modelle", "tokenizer.json");
  }

  normalisiere(text) {
    if (typeof text !== "string") {
      throw new TypeError("Der Text muss eine Zeichenkette sein.");
    }

    return text.normalize("NFC").trim();
  }

  zerlege(text) {
    const normalisiert = this.normalisiere(text);

    if (!normalisiert) {
      return [];
    }

    return normalisiert.match(
      /[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*|\p{N}+(?:[.,]\p{N}+)*|[^\s]/gu
    ) || [];
  }

  _registriereToken(token) {
    if (typeof token !== "string" || token.length === 0) {
      throw new TypeError("Ungültiges Token.");
    }

    if (this.tokenZuId.has(token)) {
      return this.tokenZuId.get(token);
    }

    const id = this.idZuToken.length;

    this.tokenZuId.set(token, id);
    this.idZuToken.push(token);

    return id;
  }

  lerneTexte(textsammlung) {
    if (!Array.isArray(textsammlung)) {
      throw new TypeError("Die Textsammlung muss ein Array sein.");
    }

    for (const text of textsammlung) {
      for (const token of this.zerlege(text)) {
        this._registriereToken(token);
      }
    }

    return this.status();
  }

  kodieren(text, optionen = {}) {
    const {
      startToken = false,
      endeToken = false
    } = optionen;

    const ids = this.zerlege(text).map(token => {
      return this.tokenZuId.has(token)
        ? this.tokenZuId.get(token)
        : this.tokenZuId.get("<UNK>");
    });

    if (startToken) {
      ids.unshift(this.tokenZuId.get("<BOS>"));
    }

    if (endeToken) {
      ids.push(this.tokenZuId.get("<EOS>"));
    }

    return ids;
  }

  dekodieren(ids, optionen = {}) {
    const {
      spezialTokensAnzeigen = false
    } = optionen;

    if (!Array.isArray(ids)) {
      throw new TypeError("Die Token-IDs müssen ein Array sein.");
    }

    const tokens = ids.map(id => {
      if (!Number.isInteger(id) || id < 0 || id >= this.idZuToken.length) {
        return "<UNK>";
      }

      return this.idZuToken[id];
    });

    const gefiltert = spezialTokensAnzeigen
      ? tokens
      : tokens.filter(token => !SPEZIAL_TOKENS.includes(token));

    return gefiltert
      .join(" ")
      .replace(/\s+([.,!?;:%)\]}»])/g, "$1")
      .replace(/([([{«])\s+/g, "$1");
  }

  idFuer(token) {
    return this.tokenZuId.has(token)
      ? this.tokenZuId.get(token)
      : this.tokenZuId.get("<UNK>");
  }

  tokenFuer(id) {
    if (!Number.isInteger(id) || id < 0 || id >= this.idZuToken.length) {
      return "<UNK>";
    }

    return this.idZuToken[id];
  }

  status() {
    return {
      anzahlTokens: this.idZuToken.length,
      anzahlSpezialTokens: SPEZIAL_TOKENS.length,
      dateipfad: Tokenizer.standardDateipfad
    };
  }

  speichern(dateipfad = Tokenizer.standardDateipfad) {
    const ordner = path.dirname(dateipfad);

    fs.mkdirSync(ordner, { recursive: true });

    const daten = {
      version: 1,
      spezialTokens: SPEZIAL_TOKENS,
      tokens: this.idZuToken
    };

    fs.writeFileSync(
      dateipfad,
      JSON.stringify(daten, null, 2),
      "utf8"
    );

    return dateipfad;
  }

  static laden(dateipfad = Tokenizer.standardDateipfad) {
    if (!fs.existsSync(dateipfad)) {
      throw new Error(
        `Tokenizer-Datei nicht gefunden: ${dateipfad}`
      );
    }

    const daten = JSON.parse(
      fs.readFileSync(dateipfad, "utf8")
    );

    if (
      daten.version !== 1 ||
      !Array.isArray(daten.tokens) ||
      !Array.isArray(daten.spezialTokens)
    ) {
      throw new Error("Ungültiges Tokenizer-Dateiformat.");
    }

    const tokenizer = new Tokenizer();

    tokenizer.tokenZuId.clear();
    tokenizer.idZuToken = [];

    for (const token of daten.tokens) {
      if (typeof token !== "string" || tokenizer.tokenZuId.has(token)) {
        throw new Error("Ungültiges oder doppeltes Token im Vokabular.");
      }

      tokenizer._registriereToken(token);
    }

    for (let i = 0; i < SPEZIAL_TOKENS.length; i++) {
      if (tokenizer.idZuToken[i] !== SPEZIAL_TOKENS[i]) {
        throw new Error("Die Spezial-Tokens sind ungültig.");
      }
    }

    return tokenizer;
  }
}

module.exports = { Tokenizer };
