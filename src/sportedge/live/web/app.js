const state = { games: [], selected: null, filter: "all", history: [], historyFilter: "all", whaleFlow: [], flowConfirm: false, view: "overview", timer: null, pollSeconds: 5 };
const $ = (id) => document.getElementById(id);
const money = (n) => `${n < 0 ? "-" : ""}$${Math.abs(Number(n || 0)).toFixed(2)}`;
const pct = (n) => n == null ? "—" : `${(Number(n) * 100).toFixed(1)}%`;
const safe = (v, fallback = "—") => (v === null || v === undefined || v === "") ? fallback : String(v);

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
  return payload;
}
function toast(message) { const el = $("toast"); el.textContent = message; el.classList.add("show"); clearTimeout(el._t); el._t = setTimeout(() => el.classList.remove("show"), 3200); }
function setConnection(ok) { const el = $("connection"); el.classList.toggle("online", ok); el.lastChild.textContent = ok ? " online" : " reconnecting"; }
function renderHeader(account, stats) {
  $("header-balance").textContent = account?.connected ? money(account.balance) : account?.configured ? "Connection error" : "Not connected";
  $("header-pnl").textContent = money(stats?.total_pnl || 0); $("header-pnl").className = stats?.total_pnl > 0 ? "positive" : stats?.total_pnl < 0 ? "negative" : "";
  $("header-winrate").textContent = stats?.win_rate == null ? "—" : pct(stats.win_rate);
}
function renderMode(data) {
  const mode = data.trading_mode || "paper"; const live = mode === "live";
  $("mode").textContent = live ? "DEMO LIVE TRADING" : "PAPER TRADING"; $("mode").classList.toggle("live", live);
  const armed = Boolean(data.strategy_enabled ?? data.paper_enabled); $("paper-toggle").checked = armed;
  const power = $("trading-power"); power.classList.toggle("on", armed); power.setAttribute("aria-pressed", String(armed));
  $("strategy-label").textContent = armed ? (live ? "DEMO TRADING ON" : "PAPER TRADING ON") : "TRADING OFF";
}
function setView(view) {
  state.view = view;
  document.querySelectorAll(".view-tab").forEach(button => button.classList.toggle("active", button.dataset.view === view));
  document.querySelectorAll(".overview-view").forEach(element => element.classList.toggle("hidden", view !== "overview"));
  $("trade-radar").classList.toggle("hidden", view !== "radar");
  $("positions-view").classList.toggle("hidden", view !== "positions");
  $("stats-view").classList.toggle("hidden", view !== "stats");
  $("history-view").classList.toggle("hidden", view !== "history");
  $("whales-view").classList.toggle("hidden", view !== "whales");
}

function renderGames() {
  const list = $("games"); list.replaceChildren();
  const games = state.games.filter(g => state.filter === "all" || g.status === state.filter);
  if (!games.length) { const p = document.createElement("p"); p.className = "empty-list"; p.textContent = "No games in this view."; list.append(p); return; }
  games.forEach(game => {
    const button = document.createElement("button"); button.className = `game-card${state.selected === game.event_id ? " selected" : ""}`;
    const top = document.createElement("div"); top.className = "game-top";
    const status = document.createElement("span"); status.className = game.is_live ? "live-text" : ""; status.textContent = game.is_live ? "● LIVE" : game.status === "post" ? "FINAL" : "UPCOMING";
    const sport = document.createElement("span"); sport.textContent = game.league; top.append(status, sport);
    const match = document.createElement("div"); match.className = "match"; match.textContent = game.display;
    const detail = document.createElement("div"); detail.className = "game-top"; detail.style.marginTop = "8px"; detail.textContent = game.short_detail || "Schedule pending";
    button.append(top, match, detail); button.onclick = () => selectGame(game.event_id); list.append(button);
  });
}

async function loadBootstrap(force = false) {
  try {
    const data = await api(`/api/bootstrap${force ? `?t=${Date.now()}` : ""}`);
    state.games = data.games; state.pollSeconds = Math.max(2, data.poll_seconds || 5); state.selected = data.selected_event_id;
    state.history = data.history || []; state.flowConfirm = Boolean(data.flow_confirm_enabled); renderMode(data); renderHeader(data.account, data.stats); renderHistory(); renderWhales(data); renderGames(); setConnection(true);
    if (state.selected) { $("empty-state").classList.add("hidden"); $("dashboard").classList.remove("hidden"); pollSnapshot(); }
  } catch (error) { setConnection(false); toast(error.message); }
}

async function selectGame(eventId) {
  try {
    state.selected = eventId; renderGames(); toast("Connecting game, model, and market feeds…");
    await api("/api/select", { method: "POST", body: JSON.stringify({ event_id: eventId }) });
    $("empty-state").classList.add("hidden"); $("dashboard").classList.remove("hidden");
    clearTimeout(state.timer); await pollSnapshot();
  } catch (error) { toast(error.message); }
}

function outcomeNode(row) {
  const article = document.createElement("article"); article.className = "outcome";
  const identity = document.createElement("div"); const title = document.createElement("h3"); title.textContent = row.label; const ticker = document.createElement("div"); ticker.className = "ticker"; ticker.textContent = row.ticker; identity.append(title, ticker);
  const prob = document.createElement("div"); prob.className = "prob"; const bar = document.createElement("div"); bar.className = "prob-bar"; const fill = document.createElement("i"); fill.style.width = `${Math.max(0, Math.min(100, row.model_probability * 100))}%`; bar.append(fill); const copy = document.createElement("div"); copy.className = "prob-copy"; copy.innerHTML = `<span>MODEL ${pct(row.model_probability)}</span><span>${safe(row.trend, "NO TREND")}</span>`; prob.append(bar, copy);
  const quote = document.createElement("div"); quote.className = "quote"; const quoteStrong = document.createElement("strong"); quoteStrong.textContent = pct(row.market_price); const quoteLabel = document.createElement("span"); quoteLabel.textContent = row.market_price == null ? "NO QUOTE" : `BID ${pct(row.bid)} · ASK ${pct(row.ask)}`; quote.append(quoteStrong, quoteLabel);
  const edge = document.createElement("div"); edge.className = "edge"; const edgeStrong = document.createElement("strong"); edgeStrong.textContent = row.edge == null ? "—" : `${row.edge >= 0 ? "+" : ""}${pct(row.edge)}`; edgeStrong.className = row.edge > 0 ? "positive" : row.edge < 0 ? "negative" : ""; const edgeLabel = document.createElement("span"); edgeLabel.textContent = "EDGE"; edge.append(edgeStrong, edgeLabel);
  const signal = document.createElement("div"); signal.className = "signal"; const sig = document.createElement("b"); sig.className = row.signal.includes("BUY") ? "buy" : ""; sig.textContent = row.signal; const reason = document.createElement("span"); reason.textContent = row.signal_reason; signal.append(sig, reason);
  article.append(identity, prob, quote, edge, signal); return article;
}

function radarRow(row) {
  const line = document.createElement("div"); line.className = "radar-table radar-row";
  const contract = document.createElement("div"); contract.className = "radar-contract"; const label = document.createElement("b"); label.textContent = row.label; const ticker = document.createElement("small"); ticker.textContent = row.ticker; contract.append(label, ticker);
  const model = document.createElement("strong"); model.textContent = pct(row.model_probability);
  const market = document.createElement("strong"); market.textContent = pct(row.market_price);
  const edge = document.createElement("strong"); edge.textContent = row.edge == null ? "—" : `${row.edge >= 0 ? "+" : ""}${pct(row.edge)}`; edge.className = ["WATCHING BOTTOM", "TRIGGERED"].includes(row.tracking_state) ? "positive" : row.edge < 0 ? "negative" : "";
  const status = document.createElement("div"); status.className = "radar-state"; const title = document.createElement("b"); title.textContent = row.tracking_state; title.className = row.tracking_state === "TRIGGERED" ? "triggered" : row.tracking_state === "WATCHING BOTTOM" ? "watching" : ""; const reason = document.createElement("small"); reason.textContent = row.tracking_reason; status.append(title, reason);
  line.append(contract, model, market, edge, status); return line;
}

function renderRadar(data) {
  const tracked = data.outcomes.filter(row => row.is_tracked).length;
  const quoted = data.outcomes.filter(row => row.market_price != null).length;
  const qualified = data.outcomes.filter(row => row.edge != null && row.edge >= data.min_edge).length;
  const triggered = data.outcomes.filter(row => row.tracking_state === "TRIGGERED").length;
  $("radar-count").textContent = tracked; $("tracked-count").textContent = tracked; $("quoted-count").textContent = quoted; $("qualified-count").textContent = qualified; $("trigger-count").textContent = triggered;
  const armed = Boolean(data.strategy_enabled ?? data.paper_enabled); const live = data.trading_mode === "live"; const arm = document.querySelector(".radar-arm"); arm.classList.toggle("armed", armed); $("radar-arm-title").textContent = armed ? (live ? "DEMO LIVE ARMED" : "PAPER ARMED") : "OBSERVE ONLY"; $("radar-arm-copy").textContent = armed ? (live ? "Bottom triggers can submit demo orders" : "Bottom triggers can write paper fills") : "Strategy is disarmed";
  const rows = $("radar-rows"); rows.replaceChildren();
  if (!data.outcomes.length) { const empty = document.createElement("div"); empty.className = "radar-empty"; empty.textContent = "No outcomes are currently being evaluated."; rows.append(empty); }
  else rows.append(...data.outcomes.map(radarRow));
}

function positionRow(position) {
  const line = document.createElement("div"); line.className = "positions-table position-row";
  const contract = document.createElement("div"); contract.className = "position-contract";
  const name = document.createElement("b"); name.textContent = position.selected_team || position.ticker;
  const ticker = document.createElement("small"); ticker.textContent = `${position.ticker}${position.matchup ? ` · ${position.matchup}` : ""}`;
  const status = document.createElement("span"); status.className = "position-status"; status.textContent = `${position.status} · ${position.fill_count} FILL${position.fill_count === 1 ? "" : "S"}`; contract.append(name, ticker, status);
  const side = document.createElement("span"); side.className = "position-side"; side.textContent = position.side;
  const shares = document.createElement("strong"); shares.textContent = Number(position.shares).toFixed(2);
  const entry = document.createElement("strong"); entry.textContent = pct(position.average_entry);
  const mark = document.createElement("strong"); mark.textContent = pct(position.mark); if (position.mark == null) mark.className = "position-unmarked";
  const cost = document.createElement("strong"); cost.textContent = money(position.cost_basis);
  const value = document.createElement("strong"); value.textContent = position.market_value == null ? "—" : money(position.market_value); if (position.market_value == null) value.className = "position-unmarked";
  const pnl = document.createElement("strong"); pnl.textContent = position.unrealized_pnl == null ? "—" : money(position.unrealized_pnl); pnl.className = position.unrealized_pnl > 0 ? "positive" : position.unrealized_pnl < 0 ? "negative" : "position-unmarked";
  line.append(contract, side, shares, entry, mark, cost, value, pnl); return line;
}

function renderPositions(data) {
  const positions = data.positions || [];
  const shares = positions.reduce((sum, row) => sum + Number(row.shares || 0), 0);
  const cost = positions.reduce((sum, row) => sum + Number(row.cost_basis || 0), 0);
  const marked = positions.filter(row => row.market_value != null);
  const value = marked.reduce((sum, row) => sum + Number(row.market_value || 0), 0);
  const pnl = marked.reduce((sum, row) => sum + Number(row.unrealized_pnl || 0), 0);
  $("positions-count").textContent = positions.length; $("held-contracts").textContent = positions.length; $("held-shares").textContent = shares.toFixed(2); $("held-cost").textContent = money(cost); $("held-value").textContent = marked.length ? money(value) : "—"; $("held-pnl").textContent = marked.length ? money(pnl) : "—"; $("held-pnl").className = pnl > 0 ? "positive" : pnl < 0 ? "negative" : "";
  const rows = $("position-rows"); rows.replaceChildren();
  if (!positions.length) { const empty = document.createElement("div"); empty.className = "radar-empty"; empty.textContent = "No open paper contracts are currently held."; rows.append(empty); }
  else rows.append(...positions.map(positionRow));
}

function renderStats(data) {
  const stats = data.stats;
  const color = (element, value) => { element.className = value > 0 ? "positive" : value < 0 ? "negative" : ""; };
  $("stats-total-pnl").textContent = money(stats.total_pnl); color($("stats-total-pnl"), stats.total_pnl);
  $("stats-return").textContent = `${pct(stats.total_return)} total return`;
  $("stats-fills").textContent = stats.fills; $("stats-avg-stake").textContent = `${money(stats.average_stake)} average stake`;
  $("stats-contracts").textContent = stats.contracts_held; $("stats-marked").textContent = `${stats.marked_positions} currently marked`;
  $("stats-record").textContent = `${stats.winners}–${stats.losers}`; $("stats-win-rate").textContent = stats.win_rate == null ? "No settled fills" : `${pct(stats.win_rate)} settled win rate`;
  $("stats-edge").textContent = pct(stats.average_edge); $("stats-exposure").textContent = money(stats.open_exposure); $("stats-staked").textContent = money(stats.total_staked);
  $("stats-realized").textContent = money(stats.realized_pnl); color($("stats-realized"), stats.realized_pnl);
  $("stats-unrealized").textContent = money(stats.unrealized_pnl); color($("stats-unrealized"), stats.unrealized_pnl);
  $("stats-realized-roi").textContent = pct(stats.realized_roi); color($("stats-realized-roi"), stats.realized_roi);
  $("stats-updated").textContent = `P&L marked ${new Date(data.pnl_updated_at * 1000).toLocaleTimeString()} · Kalshi holdings refresh every 10 seconds.`;
}

function historyRow(trade) {
  const row = document.createElement("div"); row.className = "history-table history-row";
  const time = document.createElement("div"); time.className = "history-time"; const date = new Date(trade.ts * 1000); const dateText = document.createElement("b"); dateText.textContent = date.toLocaleDateString([], { month: "short", day: "numeric" }); const clock = document.createElement("small"); clock.textContent = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" }); const mode = document.createElement("small"); mode.className = `history-mode${trade.mode === "demo-live" ? " live" : ""}`; mode.textContent = trade.mode; time.append(dateText, clock, mode);
  const contract = document.createElement("div"); contract.className = "history-contract"; const selection = document.createElement("b"); selection.textContent = trade.selection || trade.ticker; const ticker = document.createElement("small"); ticker.textContent = trade.ticker; const matchup = document.createElement("small"); matchup.textContent = trade.matchup || (trade.event_id ? `Event ${trade.event_id}` : "No game metadata"); contract.append(selection, ticker, matchup);
  const side = document.createElement("strong"); side.className = "position-side"; side.textContent = trade.side;
  const stake = document.createElement("strong"); stake.textContent = money(trade.stake);
  const price = document.createElement("strong"); price.textContent = pct(trade.price);
  const shares = document.createElement("strong"); shares.textContent = Number(trade.shares || 0).toFixed(2);
  const edge = document.createElement("strong"); edge.textContent = `${trade.edge >= 0 ? "+" : ""}${pct(trade.edge)}`; edge.className = trade.edge > 0 ? "positive" : trade.edge < 0 ? "negative" : "";
  const result = document.createElement("div"); result.className = "history-result"; const status = document.createElement("b"); status.textContent = trade.status; const detail = document.createElement("small"); detail.textContent = trade.pnl == null ? (trade.filled_count == null ? "" : `${trade.filled_count}/${trade.requested_count} filled`) : `${money(trade.pnl)} P&L`; detail.className = trade.pnl > 0 ? "positive" : trade.pnl < 0 ? "negative" : ""; result.append(status, detail); if (trade.order_id) { const order = document.createElement("small"); order.className = "history-order"; order.title = trade.order_id; order.textContent = trade.order_id; result.append(order); }
  row.append(time, contract, side, stake, price, shares, edge, result); return row;
}

function renderHistory() {
  const all = state.history || []; const visible = all.filter(trade => state.historyFilter === "all" || trade.mode === state.historyFilter);
  $("history-count").textContent = all.length; $("history-total").textContent = all.length; $("history-paper").textContent = all.filter(t => t.mode === "paper").length; $("history-live").textContent = all.filter(t => t.mode === "demo-live").length; $("history-stake").textContent = money(all.reduce((sum, trade) => sum + Number(trade.stake || 0), 0));
  const rows = $("history-rows"); rows.replaceChildren();
  if (!visible.length) { const empty = document.createElement("div"); empty.className = "history-empty"; empty.textContent = state.historyFilter === "all" ? "No trades have been recorded yet." : `No ${state.historyFilter} trades have been recorded.`; rows.append(empty); }
  else rows.append(...visible.map(historyRow));
}

function whaleMarketNode(market) {
  const card = document.createElement("article"); card.className = "whale-market"; const title = document.createElement("h3"); title.textContent = market.label; const ticker = document.createElement("code"); ticker.textContent = market.ticker;
  const assessment = market.model_assessment || { verdict: "NO WHALE", score: 0, reason: "No model assessment" }; const model = document.createElement("div"); model.className = "whale-model"; const verdict = document.createElement("b"); verdict.className = assessment.verdict.startsWith("FOLLOW") ? "follow" : assessment.verdict.startsWith("FADE") ? "fade" : ""; verdict.textContent = assessment.verdict; const score = document.createElement("strong"); score.textContent = `${assessment.score}/100`; const bar = document.createElement("i"); bar.style.setProperty("--score", `${assessment.score}%`); const why = document.createElement("small"); why.textContent = assessment.reason; model.append(verdict, score, bar, why);
  const signal = document.createElement("div"); signal.className = "whale-signal"; const flag = document.createElement("b"); flag.className = market.signal.confirms_buy ? "confirm" : ""; flag.textContent = market.signal.confirms_buy ? "CONFIRMS YES ENTRY" : "NO YES CONFIRM"; const reason = document.createElement("span"); reason.textContent = market.signal.reason; signal.append(flag, reason); card.append(title, ticker, model, signal); return card;
}
function whaleTradeNode(trade) {
  const row = document.createElement("div"); row.className = "whale-table whale-row"; const time = document.createElement("span"); time.textContent = new Date(trade.ts * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" }); const contract = document.createElement("span"); contract.textContent = trade.label; contract.title = trade.ticker; const side = document.createElement("strong"); side.textContent = trade.side; const price = document.createElement("strong"); price.textContent = pct(trade.price); const count = document.createElement("strong"); count.textContent = Number(trade.count).toFixed(2); const notional = document.createElement("strong"); notional.textContent = money(trade.notional); const flag = document.createElement("span"); flag.className = `whale-flag${trade.is_whale ? " yes" : ""}`; flag.textContent = trade.is_whale ? "WHALE" : "TRADE"; row.append(time, contract, side, price, count, notional, flag); return row;
}
function renderWhales(data = {}) {
  if (data.whale_flow) state.whaleFlow = data.whale_flow; if (data.flow_confirm_enabled !== undefined) state.flowConfirm = Boolean(data.flow_confirm_enabled);
  const markets = state.whaleFlow || []; const trades = markets.flatMap(market => market.trades.map(trade => ({ ...trade, label: market.label, ticker: market.ticker }))).sort((a,b) => b.ts - a.ts).slice(0, 200); const whales = trades.filter(trade => trade.is_whale);
  $("whale-count").textContent = whales.length; $("whale-prints").textContent = whales.length; $("whale-notional").textContent = money(markets.reduce((sum, market) => sum + Number(market.signal.notional || 0), 0)); $("whale-confirms").textContent = markets.filter(market => market.signal.confirms_buy).length; $("whale-clusters").textContent = markets.filter(market => market.signal.cluster).length;
  const confirm = $("flow-confirm"); confirm.classList.toggle("on", state.flowConfirm); confirm.setAttribute("aria-pressed", String(state.flowConfirm)); confirm.querySelector("span").textContent = state.flowConfirm ? "FLOW CONFIRMATION ON" : "FLOW CONFIRMATION OFF";
  const live = $("mode").classList.contains("live"); $("flow-mode-label").textContent = state.flowConfirm ? `Gating ${live ? "Demo Live" : "Paper"} entries` : "Observing only"; $("flow-mode-copy").textContent = state.flowConfirm ? "Model entries require a whale print or contrarian sell-off confirmation." : `Whale flow is visible but does not gate ${live ? "Demo Live" : "Paper"} Trading entries.`;
  const cards = $("whale-markets"); cards.replaceChildren(); if (!markets.length) { const empty = document.createElement("div"); empty.className = "whale-empty"; empty.textContent = "No configured contracts are available for whale tracking in this game."; cards.append(empty); } else cards.append(...markets.map(whaleMarketNode));
  const rows = $("whale-trades"); rows.replaceChildren(); if (!trades.length) { const empty = document.createElement("div"); empty.className = "whale-empty"; empty.textContent = "No recent trades in the configured lookback window."; rows.append(empty); } else rows.append(...trades.map(whaleTradeNode));
}

function renderSnapshot(data) {
  const g = data.game; $("game-status").textContent = g.status_label; $("game-status").className = `live-pill ${g.status}`;
  $("league").textContent = safe(g.league).toUpperCase(); $("clock").textContent = g.clock_label; $("stale").classList.toggle("hidden", !data.stale);
  $("away-team").textContent = g.away_team; $("home-team").textContent = g.home_team; $("away-score").textContent = g.away_score; $("home-score").textContent = g.home_score; $("last-play").textContent = safe(g.last_play_text, "No play description available.");
  const outcomes = [...data.outcomes].sort((a,b) => b.model_probability - a.model_probability); const leader = outcomes[0]; $("model-leader").textContent = leader?.label || "—"; $("model-prob").textContent = leader ? `${pct(leader.model_probability)} model probability` : "—";
  const quoted = data.outcomes.filter(o => o.edge != null).sort((a,b) => b.edge - a.edge); const best = quoted[0]; $("best-edge").textContent = best ? `${best.edge >= 0 ? "+" : ""}${pct(best.edge)}` : "—"; $("best-edge").className = best?.edge > 0 ? "positive" : best?.edge < 0 ? "negative" : ""; $("edge-label").textContent = best?.label || "No quoted market";
  $("open-exposure").textContent = money(data.ledger.open_exposure); $("open-positions").textContent = `${data.ledger.open_positions} open position${data.ledger.open_positions === 1 ? "" : "s"}`; $("total-pnl").textContent = money(data.ledger.total_pnl); $("total-pnl").className = data.ledger.total_pnl > 0 ? "positive" : data.ledger.total_pnl < 0 ? "negative" : ""; $("fill-count").textContent = `${data.ledger.fills} paper fills`;
  const board = $("outcomes"); board.replaceChildren(...data.outcomes.map(outcomeNode));
  renderRadar(data);
  renderPositions(data);
  renderStats(data);
  state.history = data.history || []; renderHistory();
  renderWhales(data);
  const context = $("context"); context.replaceChildren(); const items = g.sport === "soccer" ? [["Minute", g.clock_label],["Possession", safe(g.possession,"Unknown")],["Yellow cards", `${g.away_yellow} — ${g.home_yellow}`],["Red cards", `${g.away_red} — ${g.home_red}`],["Set piece", safe(g.set_piece,"None")]] : [["Period", g.period],["Clock", g.clock_label],["Possession", safe(g.possession,"Unknown")],["Team fouls", `${g.away_fouls} — ${g.home_fouls}`],["Free throw", g.free_throw_active ? "Active" : "No"]]; items.forEach(([k,v]) => { const dt=document.createElement("dt"), dd=document.createElement("dd"); dt.textContent=k; dd.textContent=v; context.append(dt,dd); });
  $("updated").textContent = `Updated ${new Date(data.updated_at * 1000).toLocaleTimeString()}`; renderMode(data);
  renderHeader(data.account, data.stats);
}

async function pollSnapshot() {
  if (!state.selected) return;
  try { const data = await api("/api/snapshot"); renderSnapshot(data); setConnection(true); }
  catch (error) { setConnection(false); toast(error.message); }
  finally { clearTimeout(state.timer); state.timer = setTimeout(pollSnapshot, state.pollSeconds * 1000); }
}

document.querySelectorAll(".filter").forEach(button => button.onclick = () => { document.querySelectorAll(".filter").forEach(b => b.classList.remove("active")); button.classList.add("active"); state.filter = button.dataset.filter; renderGames(); });
document.querySelectorAll(".view-tab").forEach(button => button.onclick = () => setView(button.dataset.view));
document.querySelectorAll(".history-filter").forEach(button => button.onclick = () => { document.querySelectorAll(".history-filter").forEach(item => item.classList.remove("active")); button.classList.add("active"); state.historyFilter = button.dataset.historyFilter; renderHistory(); });
$("refresh-games").onclick = () => loadBootstrap(true);
async function setTradingPower(enabled) {
  const live = $("mode").classList.contains("live");
  if (enabled && live && !window.confirm("Turn demo trading ON? Qualifying signals can submit orders using demo funds.")) return;
  try { const data = await api("/api/paper", { method: "POST", body: JSON.stringify({ enabled }) }); renderMode({ ...data, trading_mode: live ? "live" : "paper" }); toast(data.strategy_enabled ? `${live ? "Demo" : "Paper"} trading is ON.` : "Trading is OFF."); }
  catch (error) { toast(error.message); }
}
$("trading-power").onclick = () => setTradingPower(!$("paper-toggle").checked);
$("mode").onclick = async () => {
  const next = $("mode").classList.contains("live") ? "paper" : "live";
  if (next === "live" && !window.confirm("Switch to Demo Live Trading? The strategy will be disarmed. If you arm it afterward, qualifying signals can submit orders using demo funds.")) return;
  try { const data = await api("/api/mode", { method: "POST", body: JSON.stringify({ mode: next }) }); renderMode(data); renderWhales(); toast(`${data.label} selected. Strategy is disarmed.`); }
  catch (error) { toast(error.message); }
};
$("flow-confirm").onclick = async () => { try { const data = await api("/api/flow", { method: "POST", body: JSON.stringify({ enabled: !state.flowConfirm }) }); state.flowConfirm = data.flow_confirm_enabled; renderWhales(data); toast(state.flowConfirm ? "Whale flow now gates model entries." : "Whale flow is observation-only."); } catch (error) { toast(error.message); } };

async function openSettings() {
  $("settings-modal").classList.remove("hidden");
  try {
    const data = await api("/api/settings");
    $("setting-host").value = data.kalshi_host;
    $("setting-key-id").value = ""; $("setting-key-id").placeholder = data.api_key_configured ? `Configured ${data.api_key_hint} — leave blank to keep` : "Paste your Kalshi API key ID";
    $("setting-private-key").value = ""; $("setting-private-key").placeholder = data.private_key_configured ? "Private key configured — leave blank to keep it" : "-----BEGIN PRIVATE KEY-----\nPaste your private key here\n-----END PRIVATE KEY-----";
    renderCredentialStatus(data);
  } catch (error) { toast(error.message); }
}
function closeSettings() { $("settings-modal").classList.add("hidden"); }
function renderCredentialStatus(data) {
  const status = $("credential-status"); const connected = data.account?.connected; status.classList.toggle("connected", connected);
  status.querySelector("b").textContent = connected ? `Connected · ${money(data.account.balance)} available` : data.api_key_configured && data.private_key_configured ? "Credentials saved · connection unavailable" : "Kalshi account not connected";
  status.querySelector("small").textContent = connected ? "Read-only balance check succeeded." : "Add both credentials and verify the API host.";
  $("header-balance").textContent = connected ? money(data.account.balance) : data.account?.configured ? "Connection error" : "Not connected";
}
$("settings-open").onclick = openSettings;
document.querySelectorAll("[data-close-settings]").forEach(element => element.onclick = closeSettings);
$("settings-form").onsubmit = async (event) => {
  event.preventDefault(); const save = $("settings-save"); save.disabled = true; save.textContent = "Testing read-only connection…";
  try {
    const data = await api("/api/settings", { method: "POST", body: JSON.stringify({ kalshi_host: $("setting-host").value, api_key_id: $("setting-key-id").value, private_key_pem: $("setting-private-key").value }) });
    renderCredentialStatus(data); $("setting-key-id").value = ""; $("setting-private-key").value = ""; toast("Kalshi account connected and saved locally.");
  } catch (error) { toast(error.message); }
  finally { save.disabled = false; save.textContent = "Test connection & save"; }
};
loadBootstrap();
