"""Kelly criterion bet simulator on historical predictions vs market odds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src import config


def load_predictions_for_season(show_slug: str, season: int) -> list[dict]:
    """Load all per-episode prediction files, sorted by episode."""
    pred_dir = config.predictions_dir(show_slug, season)
    if not pred_dir.exists():
        return []
    files = sorted(pred_dir.glob("episode_*.json"))
    return [json.loads(f.read_text()) for f in files]


def load_market_snapshots(show_slug: str, season: int) -> dict[int, dict[str, float]]:
    """
    Load per-episode Kalshi market snapshots.
    Returns { episode_number: { contestant_name: implied_prob } }
    """
    market_dir = config.market_dir(show_slug, season)
    snapshots: dict[int, dict[str, float]] = {}
    if not market_dir.exists():
        return snapshots
    for path in sorted(market_dir.glob("episode_*.json")):
        with open(path) as f:
            data = json.load(f)
        ep = data.get("episode_number")
        if ep is not None:
            snapshots[ep] = data.get("winner_odds", {})
    return snapshots


def load_outcomes(show_slug: str, season: int) -> dict:
    path = config.season_dir(show_slug, season) / "outcomes.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def kelly_fraction(model_prob: float, market_prob: float, kelly_frac: float = 0.25) -> float:
    """
    Compute the fractional Kelly bet size (as a fraction of bankroll).

    f* = (edge / odds) * kelly_frac
    where:
      edge  = model_prob - market_prob
      odds  = (1 - market_prob) / market_prob   (decimal odds - 1 for a "Yes" bet)

    Returns 0 if the edge is negative or zero.
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    payout = (1 - market_prob) / market_prob  # net payout per unit wagered
    full_kelly = (model_prob * payout - (1 - model_prob)) / payout
    return max(0.0, full_kelly * kelly_frac)


def simulate(
    show_slug: str,
    season: int,
    starting_bankroll: float = config.SIM_STARTING_BANKROLL,
    edge_threshold: float = config.SIM_EDGE_THRESHOLD,
    kelly_frac: float = config.SIM_KELLY_FRACTION,
) -> dict:
    """
    Run Kelly criterion simulation across all episodes for a season.

    For each episode where |model_prob - market_prob| > edge_threshold,
    simulate a Kalshi "Yes" (model > market) or "No" (model < market) trade.

    Assumes winner market (contestant-level win probability contracts).
    """
    predictions_by_ep = {p["episode_number"]: p for p in load_predictions_for_season(show_slug, season)}
    market_by_ep = load_market_snapshots(show_slug, season)
    outcomes = load_outcomes(show_slug, season)
    winner = outcomes.get("winner")

    bankroll = starting_bankroll
    trades: list[dict] = []
    bankroll_history: list[dict] = []

    for ep_num in sorted(predictions_by_ep.keys()):
        pred = predictions_by_ep[ep_num]
        market = market_by_ep.get(ep_num, {})
        if not market:
            bankroll_history.append({"episode": ep_num, "bankroll": bankroll, "trades_this_ep": 0})
            continue

        ep_trades = []
        for name, data in pred.get("contestants", {}).items():
            model_p = data.get("win_prob", 0.0)
            market_p = market.get(name)
            if market_p is None:
                continue

            delta = model_p - market_p
            if abs(delta) < edge_threshold:
                continue

            # Bet Yes if model > market, No if model < market
            position = "yes" if delta > 0 else "no"
            if position == "yes":
                bet_model_p = model_p
                bet_market_p = market_p
            else:
                # Bet No: model thinks they're LESS likely to win
                bet_model_p = 1 - model_p
                bet_market_p = 1 - market_p

            frac = kelly_fraction(bet_model_p, bet_market_p, kelly_frac)
            if frac <= 0:
                continue

            stake = bankroll * frac

            # Settle: winner is resolved at season end — mark as pending until winner known
            settled = winner is not None
            if settled:
                won = (position == "yes" and name == winner) or (
                    position == "no" and name != winner
                )
            else:
                won = None

            payout_ratio = (1 - bet_market_p) / bet_market_p if bet_market_p > 0 else 0
            profit = stake * payout_ratio if (settled and won) else (-stake if (settled and not won) else 0)

            trade = {
                "episode": ep_num,
                "contestant": name,
                "position": position,
                "model_prob": round(model_p, 4),
                "market_prob": round(market_p, 4),
                "delta": round(delta, 4),
                "kelly_fraction": round(frac, 4),
                "stake": round(stake, 2),
                "settled": settled,
                "won": won,
                "profit": round(profit, 2),
            }
            ep_trades.append(trade)
            if settled:
                bankroll += profit

        trades.extend(ep_trades)
        bankroll_history.append({
            "episode": ep_num,
            "bankroll": round(bankroll, 2),
            "trades_this_ep": len(ep_trades),
        })

    # Compute summary stats
    settled_trades = [t for t in trades if t["settled"]]
    winning_trades = [t for t in settled_trades if t["won"]]
    losing_trades = [t for t in settled_trades if not t["won"]]

    roi_pct = ((bankroll - starting_bankroll) / starting_bankroll * 100) if starting_bankroll > 0 else 0.0

    # Max drawdown
    peak = starting_bankroll
    max_drawdown = 0.0
    running = starting_bankroll
    for h in bankroll_history:
        running = h["bankroll"]
        peak = max(peak, running)
        drawdown = (peak - running) / peak * 100 if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    best_trade = max(settled_trades, key=lambda t: t["profit"], default=None)
    worst_trade = min(settled_trades, key=lambda t: t["profit"], default=None)

    return {
        "show": show_slug,
        "season": season,
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": round(bankroll, 2),
        "roi_pct": round(roi_pct, 2),
        "total_trades": len(trades),
        "settled_trades": len(settled_trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": round(len(winning_trades) / len(settled_trades), 4) if settled_trades else None,
        "max_drawdown_pct": round(max_drawdown, 2),
        "edge_threshold": edge_threshold,
        "kelly_fraction": kelly_frac,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "bankroll_history": bankroll_history,
        "trades": trades,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kelly bet simulator")
    parser.add_argument("--show", default="survivor")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--bankroll", type=float, default=config.SIM_STARTING_BANKROLL)
    parser.add_argument("--edge-threshold", type=float, default=config.SIM_EDGE_THRESHOLD)
    parser.add_argument("--kelly-fraction", type=float, default=config.SIM_KELLY_FRACTION)
    args = parser.parse_args()

    result = simulate(
        show_slug=args.show,
        season=args.season,
        starting_bankroll=args.bankroll,
        edge_threshold=args.edge_threshold,
        kelly_frac=args.kelly_fraction,
    )

    print(f"\nBankroll: ${result['starting_bankroll']} → ${result['ending_bankroll']}")
    print(f"ROI: {result['roi_pct']}%  |  Win rate: {result['win_rate']}")
    print(f"Trades: {result['total_trades']} ({result['winning_trades']}W / {result['losing_trades']}L)")
    print(f"Max drawdown: {result['max_drawdown_pct']}%")
