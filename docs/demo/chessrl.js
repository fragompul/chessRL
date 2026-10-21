// chessRL in the browser: board encoding, action space and move selection for the ONNX network.
// A line-by-line port of src/chessrl/rl/{encoding,actions}.py — the Node test (test/) checks it
// produces bit-identical planes and the same action indices as the Python code.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.ChessRL = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const NUM_PLANES = 20, NUM_ACTIONS = 4672;
  const TYPE = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
  const DIRS = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]];
  const KNIGHT = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  const UNDERPROMO = ["n", "b", "r"];

  const sqIndex = (name) => (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97);
  const sqName = (i) => String.fromCharCode(97 + (i & 7)) + String.fromCharCode(49 + (i >> 3));

  // ----- encoding --------------------------------------------------------------------
  // `chess` is a chess.js instance; `repeated` says whether the position has occurred before.
  function encodeBoard(chess, repeated) {
    const out = new Float32Array(NUM_PLANES * 64);
    const turn = chess.turn();
    const white = turn === "w";
    const flip = (sq) => (white ? sq : sq ^ 56);
    const board = chess.board(); // board[0] is rank 8
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const p = board[row][col];
        if (!p) continue;
        const sq = flip((7 - row) * 8 + col);
        const own = (p.color === "w") === white;
        out[((own ? 0 : 6) + TYPE[p.type] - 1) * 64 + sq] = 1;
      }
    }
    const fields = chess.fen().split(" ");
    const rights = fields[2];
    const has = (c) => rights.includes(c);
    const own = white ? [has("K"), has("Q")] : [has("k"), has("q")];
    const opp = white ? [has("k"), has("q")] : [has("K"), has("Q")];
    [own[0], own[1], opp[0], opp[1]].forEach((v, i) => { if (v) out.fill(1, (12 + i) * 64, (13 + i) * 64); });
    // en passant: set after ANY double pawn push (the Python engine's convention), not only when capturable
    const hist = chess.history({ verbose: true });
    const last = hist[hist.length - 1];
    if (last && last.piece === "p" && Math.abs(sqIndex(last.to) - sqIndex(last.from)) === 16) {
      out[16 * 64 + flip((sqIndex(last.from) + sqIndex(last.to)) >> 1)] = 1;
    }
    out.fill(Math.fround(parseInt(fields[4], 10) / 100), 17 * 64, 18 * 64);
    if (repeated) out.fill(1, 18 * 64, 19 * 64);
    out.fill(1, 19 * 64, 20 * 64);
    return out;
  }

  // ----- action space ------------------------------------------------------------------
  const sign = (x) => (x > 0) - (x < 0);
  // `move` is a chess.js move object {from, to, promotion?, color}
  function moveToIndex(move) {
    let frm = sqIndex(move.from), to = sqIndex(move.to);
    if (move.color === "b") { frm ^= 56; to ^= 56; }
    const df = (to & 7) - (frm & 7), dr = (to >> 3) - (frm >> 3);
    let plane;
    if (move.promotion && move.promotion !== "q") {
      plane = 64 + (df + 1) * 3 + UNDERPROMO.indexOf(move.promotion);
    } else {
      const k = KNIGHT.findIndex(([a, b]) => a === df && b === dr);
      if (k >= 0) plane = 56 + k;
      else {
        const d = DIRS.findIndex(([a, b]) => a === sign(df) && b === sign(dr));
        plane = d * 7 + Math.max(Math.abs(df), Math.abs(dr)) - 1;
      }
    }
    return frm * 73 + plane;
  }

  // ----- move selection ---------------------------------------------------------------------
  function softmax(xs, temperature) {
    const t = Math.max(temperature, 1e-6);
    const m = Math.max(...xs);
    const e = xs.map((x) => Math.exp((x - m) / t));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function sample(probs, rng) {
    let r = rng(), acc = 0;
    for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (r <= acc) return i; }
    return probs.length - 1;
  }

  // Terminal value of a position for the side to move, or null if the game goes on.
  function terminalValue(chess) {
    if (chess.isCheckmate()) return -1;
    if (chess.isDraw() || chess.isStalemate() || chess.isInsufficientMaterial() || chess.isThreefoldRepetition()) return 0;
    return null;
  }

  return { NUM_PLANES, NUM_ACTIONS, encodeBoard, moveToIndex, softmax, sample, terminalValue, sqIndex, sqName };
});
