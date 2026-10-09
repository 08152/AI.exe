
"use strict";

// netz.js – selbst programmiertes neuronales Netz

const HIDDEN = 6;
const LEARNING_RATE = 0.5;

function zufallsgewicht() {
  return Math.random() * 2 - 1;
}

function sigmoid(x) {
  x = Math.max(-30, Math.min(30, x));
  return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
  return Math.tanh(x);
}

class NeuronalesNetz {
  constructor() {
    this.w1 = Array.from(
      { length: HIDDEN },
      () => [zufallsgewicht(), zufallsgewicht()]
    );

    this.b1 = Array.from(
      { length: HIDDEN },
      zufallsgewicht
    );

    this.w2 = Array.from(
      { length: HIDDEN },
      zufallsgewicht
    );

    this.b2 = zufallsgewicht();
    this.trainingsdurchlaeufe = 0;
    this.fehler = null;
  }

  vorhersage(x1, x2) {
    const hidden = [];

    for (let j = 0; j < HIDDEN; j++) {
      const summe =
        this.w1[j][0] * x1 +
        this.w1[j][1] * x2 +
        this.b1[j];

      hidden[j] = tanh(summe);
    }

    let summe = this.b2;

    for (let j = 0; j < HIDDEN; j++) {
      summe += this.w2[j] * hidden[j];
    }

    const ausgabe = sigmoid(summe);

    return {
      ausgabe,
      antwort: ausgabe >= 0.5 ? 1 : 0
    };
  }

  trainiereBeispiel(beispiel) {
    const { x1, x2, target } = beispiel;
    const ergebnis = this.vorhersage(x1, x2);
    const y = ergebnis.ausgabe;

    const hidden = [];

    for (let j = 0; j < HIDDEN; j++) {
      hidden[j] = tanh(
        this.w1[j][0] * x1 +
        this.w1[j][1] * x2 +
        this.b1[j]
      );
    }

    const ausgabeFehler =
      (y - target) * y * (1 - y);

    const hiddenFehler = [];

    for (let j = 0; j < HIDDEN; j++) {
      hiddenFehler[j] =
        ausgabeFehler *
        this.w2[j] *
        (1 - hidden[j] * hidden[j]);
    }

    for (let j = 0; j < HIDDEN; j++) {
      this.w2[j] -=
        LEARNING_RATE * ausgabeFehler * hidden[j];
    }

    this.b2 -= LEARNING_RATE * ausgabeFehler;

    for (let j = 0; j < HIDDEN; j++) {
      this.w1[j][0] -=
        LEARNING_RATE * hiddenFehler[j] * x1;

      this.w1[j][1] -=
        LEARNING_RATE * hiddenFehler[j] * x2;

      this.b1[j] -= LEARNING_RATE * hiddenFehler[j];
    }
  }

  trainiere(beispiele, durchlaeufe = 20000) {
    if (!Array.isArray(beispiele) || beispiele.length === 0) {
      throw new Error("Keine Trainingsdaten vorhanden.");
    }

    for (let epoche = 0; epoche < durchlaeufe; epoche++) {
      for (let i = beispiele.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [beispiele[i], beispiele[j]] =
          [beispiele[j], beispiele[i]];
      }

      for (const beispiel of beispiele) {
        this.trainiereBeispiel(beispiel);
      }
    }

    this.trainingsdurchlaeufe += durchlaeufe;

    let gesamterFehler = 0;

    for (const beispiel of beispiele) {
      const y = this.vorhersage(
        beispiel.x1,
        beispiel.x2
      ).ausgabe;

      gesamterFehler += Math.pow(y - beispiel.target, 2);
    }

    this.fehler = gesamterFehler / beispiele.length;
    return this.fehler;
  }

  status() {
    return {
      neuronen: HIDDEN,
      trainingsdurchlaeufe: this.trainingsdurchlaeufe,
      fehler: this.fehler
    };
  }
}

module.exports = { NeuronalesNetz };
