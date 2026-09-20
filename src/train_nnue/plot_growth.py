"""Render standalone growth PNG/SVG/HTML and CSV using the CPU plotting environment."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot(directory):
    directory = Path(directory)
    rows = sorted((json.loads(p.read_text()) for p in (directory / "measurements").glob("*.json")),
                  key=lambda v: v["finished_at"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].set_title("Fixed-data best-move accuracy")
    axes[1].set_title("Versus fixed baseline: wins and draw-adjusted score")
    labels = ["Accuracy (%)", "Rate (%)"]
    if rows:
        dates = [datetime.fromisoformat(v["finished_at"]) for v in rows]
        for ax, key, color, label in ((axes[0], "accuracy", "#2563eb", "Accuracy"),
                                       (axes[1], "win_rate", "#16a34a", "Wins / all games"),
                                       (axes[1], "score_rate", "#d97706", "(Wins + 0.5 draws) / all games")):
            if key == "accuracy":
                values = [r["metrics"][key]["accuracy"] for r in rows]
                low = [r["metrics"][key]["wilson_95"]["lower"] for r in rows]
                high = [r["metrics"][key]["wilson_95"]["upper"] for r in rows]
            else:
                values = [r["metrics"][key]["value"] for r in rows]
                low = [r["metrics"][key]["lower"] for r in rows]
                high = [r["metrics"][key]["upper"] for r in rows]
            ax.errorbar(dates, [100*v for v in values],
                        yerr=[[100*(v-l) for v,l in zip(values,low)],
                              [100*(h-v) for v,h in zip(values,high)]],
                        fmt="o-", capsize=4, color=color, label=label)
        axes[0].set_ylim(max(0, min(r["metrics"]["accuracy"]["wilson_95"]["lower"] for r in rows)*100-1),
                         min(100, max(r["metrics"]["accuracy"]["wilson_95"]["upper"] for r in rows)*100+1))
        axes[1].set_ylim(max(0, min(r["metrics"][k]["lower"] for r in rows for k in ("win_rate", "score_rate"))*100-2),
                         min(100, max(r["metrics"][k]["upper"] for r in rows for k in ("win_rate", "score_rate"))*100+2))
        previous = None
        model_number = 0
        for date, row in zip(dates, rows):
            model = row["champion"]["descriptor"]["id"]
            if model != previous:
                model_number += 1
                value = row["metrics"]["accuracy"]["accuracy"]*100
                offset = -18 if value > axes[0].get_ylim()[1] - 2 else 10
                axes[0].annotate(f"M{model_number}", (date, value),
                                 xytext=(5, offset), textcoords="offset points", fontsize=9)
            previous = model
        margin = max(timedelta(hours=12), (max(dates)-min(dates))*.05)
        for ax in axes:
            ax.set_xlim(min(dates)-margin, max(dates)+margin)
        axes[1].legend(fontsize=8)
    else:
        for ax in axes:
            ax.text(.5, .5, "No completed measurements yet", transform=ax.transAxes, ha="center")
    for ax, label in zip(axes, labels):
        ax.set_ylabel(label)
        if not rows:
            ax.set_ylim(0, 100)
        ax.grid(alpha=.25)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M", tz=timezone.utc))
    axes[1].set_xlabel("Measurement completion (UTC); bars = nominal 95% intervals")
    fig.savefig(directory / "growth.png", dpi=160)
    fig.savefig(directory / "growth.svg")
    plt.close(fig)
    with (directory / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["finished_at", "champion", "accuracy", "positions", "wins", "losses", "draws", "win_rate", "score_rate"])
        for row in rows:
            m = row["metrics"]
            writer.writerow([row["finished_at"], row["champion"]["descriptor"]["id"], m["accuracy"]["accuracy"],
                             m["accuracy"]["total"], m["wins"], m["losses"], m["draws"], m["win_rate"]["value"], m["score_rate"]["value"]])
    body = []
    previous, model_number = None, 0
    for row in rows:
        m = row["metrics"]
        model = row["champion"]["descriptor"]["id"]
        if model != previous:
            model_number += 1
        previous = model
        body.append("<tr>" + "".join("<td>" + html.escape(str(v)) + "</td>" for v in (
            row["finished_at"], f"M{model_number}: {model}",
            f"{m['accuracy']['accuracy']:.2%} ({m['accuracy']['matches']}/{m['accuracy']['total']})",
            f"{m['win_rate']['value']:.2%}", f"{m['score_rate']['value']:.2%}",
            f"{m['wins']} / {m['losses']} / {m['draws']}")) + "</tr>")
    protocol = json.loads((directory / "protocol.json").read_text())["config"]
    settings = html.escape(f"Series: {protocol['series']} | Baseline: {protocol['baseline']['name']} | "
                           f"Accuracy: {protocol['accuracy_nodes']} nodes | Match: {protocol['match_nodes']} nodes")
    (directory / "growth.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>研究の成長記録</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:2em auto}img{width:100%}td,th{padding:.6em;border-bottom:1px solid #ddd}</style>'
        '<h1>固定条件での成長記録</h1><p>正解率と対基準勝率。日次値は監視用で、最良候補の選抜には使いません。'
        '同じモデルの繰り返しを独立標本として合算しません。失敗・未完了は曲線に含めません。</p>'
        '<p>' + settings + '</p><img src="growth.png" alt="正解率と勝率の推移"><p><a href="growth.svg">SVG</a> · <a href="history.csv">CSV</a> · <a href="protocol.json">固定条件</a></p>'
        '<table><tr><th>完了日時 UTC</th><th>最良候補</th><th>正解率</th><th>勝/全局</th><th>引分0.5勝換算</th><th>勝/敗/分</th></tr>'
        + "".join(reversed(body)) + '</table>')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series-dir", type=Path, required=True)
    plot(parser.parse_args().series_dir)
