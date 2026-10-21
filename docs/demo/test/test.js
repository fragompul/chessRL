// Replays each fixture game with chess.js and checks planes + action indices against Python.
const crypto = require("crypto");
const { Chess } = require("chess.js");
const R = require("../chessrl.js");
const fixtures = require("./fixtures.json");

let bad = 0;
for (const [i, f] of fixtures.entries()) {
  const chess = new Chess();
  const seen = new Map();
  const key = () => chess.fen().split(" ").slice(0, 4).join(" ");
  seen.set(key(), 1);
  let repeated = false;
  for (const uci of f.moves) {
    chess.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    const k = key();
    seen.set(k, (seen.get(k) || 0) + 1);
    repeated = seen.get(k) >= 2;
  }
  const planes = R.encodeBoard(chess, repeated);
  const sha = crypto.createHash("sha256").update(Buffer.from(planes.buffer)).digest("hex");
  if (sha !== f.planes_sha256) { console.log(`case ${i}: planes differ (fen ${f.fen}, repeated ${repeated} vs ${f.repeated})`); bad++; }
  const legal = {};
  for (const m of chess.moves({ verbose: true })) legal[m.from + m.to + (m.promotion || "")] = R.moveToIndex(m);
  const a = JSON.stringify(Object.entries(legal).sort()), b = JSON.stringify(Object.entries(f.legal).sort());
  if (a !== b) { console.log(`case ${i}: action indices differ for ${f.fen}`); bad++; }
}
console.log(bad ? `FAILED: ${bad} problems in ${fixtures.length} cases` : `OK: ${fixtures.length} positions match Python exactly`);
process.exit(bad ? 1 : 0);
