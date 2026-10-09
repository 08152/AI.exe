
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const VERSION = 2;

const SPEZIAL_TOKENS = [
  "<PAD>",
  "<UNK>",
  "<BOS>",
  "<EOS>",
  "<SEP>",
  "<benutzer>",
  "<ki>"
];

class Tokenizer {
  constructor() {
    this.tokenZuId = new Map();
    this.idZuToken = [];
    this.spezialTokens = [...SPEZIAL_TOKENS];
    this.spezialTokenSet = new Set(SPEZIAL_TOKENS);

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
      throw new TypeError(
        "Der Text muss eine Zeichenkette sein."
      );
    }

    return text.normalize("NFC").trim();
  }

  // WICHTIG:
  // Markierungen wie <benutzer> und <ki> bleiben
  // jeweils ein zusammenhängendes Token.
  zerlege(text) {
    const normalisiert = this.normalisiere(text);

    if (!normalisiert) {
      return [];
    }

    const muster =
      /<[^>\s]+>|[\p{L}\p{M}]+(?:['’][\p{L}\p{M}]+)*|\p{N}+(?:[.,]\p{N}+)*|[^\s]/gu;

    return normalisiert.match(muster) || [];
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
      throw new TypeError(
        "Die Textsammlung muss ein Array sein."
      );
    }

    for (const text of textsammlung) {
      if (typeof text !== "string") {
        continue;
      }

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
      const id = this.tokenZuId.get(token);

      return id === undefined
        ? this.tokenZuId.get("<UNK>")
        : id;
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
      throw new TypeError(
        "Die Token-IDs müssen ein Array sein."
      );
    }

    let tokens = ids.map(id => {
      if (
        !Number.isInteger(id) ||
        id < 0 ||
        id >= this.idZuToken.length
      ) {
        return "<UNK>";
      }

      return this.idZuToken[id];
    });

    if (!spezialTokensAnzeigen) {
      tokens = tokens.filter(
        token => !this.spezialTokenSet.has(token)
      );
    }

    return this.formatiere(tokens);
  }

  formatiere(tokens) {
    return tokens
      .join(" ")
      .replace(/\s+([.,!?;:%)\]}»])/g, "$1")
      .replace(/([([{«])\s+/g, "$1")
      .trim();
  }

  idFuer(token) {
    const id = this.tokenZuId.get(token);

    return id === undefined
      ? this.tokenZuId.get("<UNK>")
      : id;
  }

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

  hat(token) {
    return this.tokenZuId.has(token);
  }

  status() {
    return {
      version: VERSION,
      anzahlTokens: this.idZuToken.length,
      anzahlSpezialTokens: SPEZIAL_TOKENS.length,
      spezialTokens: [...SPEZIAL_TOKENS],
      unbekanntesTokenId: this.tokenZuId.get("<UNK>")
    };
  }

  speichern(dateipfad = Tokenizer.standardDateipfad) {
    const ordner = path.dirname(dateipfad);

    fs.mkdirSync(ordner, {
      recursive: true
    });

    const daten = {
      version: VERSION,
      spezialTokens: [...SPEZIAL_TOKENS],
      tokens: [...this.idZuToken]
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
      ![1, VERSION].includes(daten.version) ||
      !Array.isArray(daten.tokens)
    ) {
      throw new Error(
        "Ungültiges Tokenizer-Dateiformat."
      );
    }

    const tokenizer = new Tokenizer();

    tokenizer.tokenZuId.clear();
    tokenizer.idZuToken = [];

    // Bestehende IDs beibehalten, damit gespeicherte
    // Vokabulare nicht unnötig durcheinandergeraten.
    for (const token of daten.tokens) {
      if (
        typeof token !== "string" ||
        token.length === 0 ||
        tokenizer.tokenZuId.has(token)
      ) {
        throw new Error(
          "Ungültiges oder doppeltes Token im Vokabular."
        );
      }

      tokenizer._registriereToken(token);
    }

    const grundTokens = [
      "<PAD>",
      "<UNK>",
      "<BOS>",
      "<EOS>",
      "<SEP>"
    ];

    for (let i = 0; i < grundTokens.length; i++) {
      if (tokenizer.idZuToken[i] !== grundTokens[i]) {
        throw new Error(
          "Die Reihenfolge der grundlegenden Spezial-Tokens ist ungültig."
        );
      }
    }

    // Fehlende Spezial-Tokens an das Ende setzen.
    // Bereits vorhandene Token-IDs bleiben unverändert.
    for (const token of SPEZIAL_TOKENS) {
      tokenizer._registriereToken(token);
    }

    tokenizer.spezialTokens = [...SPEZIAL_TOKENS];
    tokenizer.spezialTokenSet = new Set(SPEZIAL_TOKENS);

    return tokenizer;
  }
}

module.exports = {
  Tokenizer
};
