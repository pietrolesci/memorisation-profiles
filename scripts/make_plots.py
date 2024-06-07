from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotnine as pn

DATA_PATH = "./data/pile-deduped-effects/"
OUT_PATH = "./results/"


def as_cat(x: pd.Series) -> pd.Categorical:
    return pd.Categorical(x.tolist(), categories=["70M", "160M", "410M", "1.4B", "6.9B", "12B"])


def main(metric: str) -> None:
    data_path = Path(DATA_PATH)
    out_path = Path(OUT_PATH)
    out_path.mkdir(exist_ok=True, parents=True)

    # =========
    # Load data
    # =========
    df = pd.concat(
        [
            pd.read_parquet(p).assign(model_size=p.name.split("_")[1].split("-")[0].strip())
            for p in data_path.glob(f"cs*{metric}-all_boot*")
        ]
    )


    if metric == "sup_seq":
        df = (
            df
            .query("model_size != '2.8b'")
            .assign(
                model_size=lambda _df: as_cat(_df["model_size"].str.upper()),
                memorisation=lambda _df: -_df["ATT"],
                lower=lambda _df: -_df["lower"],
                upper=lambda _df: -_df["upper"],
                std_error=lambda _df: -_df["std_error"],
            )
            .assign(
                stdupper=lambda _df: _df["memorisation"] + (2 * _df["std_error"]),
                stdlower=lambda _df: _df["memorisation"] - (2 * _df["std_error"]),
            )
            .assign(
                zero_not_in_cf=lambda _df: ((_df["stdlower"] > 0) | (_df["stdupper"] < 0)).map({True: "*", False: ""})
            )
        )

    else:
        df = (
            df
            .query("model_size != '2.8b'")
            .assign(
                model_size=lambda _df: as_cat(_df["model_size"].str.upper()),
                memorisation=lambda _df: _df["ATT"],
            )
            .assign(
                stdupper=lambda _df: _df["memorisation"] + (2 * _df["std_error"]),
                stdlower=lambda _df: _df["memorisation"] - (2 * _df["std_error"]),
            )
            .assign(
                zero_not_in_cf=lambda _df: ((_df["stdlower"] > 0) | (_df["stdupper"] < 0)).map({True: "*", False: ""})
            )
        )

    # =========================
    # Memorisation profile full
    # =========================
    mem_prof = (
        pn.ggplot(
            df.query("(time >= cohort) & (zero_not_in_cband == '*')"),
            pn.aes("time", "cohort", alpha="memorisation", colour="model_size")
        ) +
        pn.geom_tile(pn.aes(width=1000, height=1000)) +
        pn.geom_vline(xintercept=95000, size=.6, colour="black", linetype="dashed", alpha=1) +
        pn.facet_wrap("model_size", ncol=3) +
        pn.labs(title="", x="Checkpoint Step", y="Treatment Step") +
        pn.scale_x_continuous(
            expand=(0, 0), 
            breaks=range(20000, 143000, 20000),
            labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 == 0 and v != 0 else "" for i, v in enumerate(x)],
        ) +
        pn.scale_y_continuous(
            expand=(0, 0), 
            breaks=range(20000, 143000, 20000), 
            labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 == 0 else "" for i, v in enumerate(x)],
        ) +
        pn.scale_alpha(range=(.5, 1.), guide=None) +
        pn.scale_colour_discrete(guide=None) +
        pn.theme_bw(base_size=11) +
        pn.theme(panel_grid_major=pn.element_blank(), plot_margin=0, plot_background=None, figure_size=(8, 3.2))
    )

    # mem_prof.save(str(out_path / f"mem_prof_{metric}_full.svg"))
    pn.save_as_pdf_pages([mem_prof], str(out_path / f"mem_prof_{metric}_full.pdf"))
    
    if metric != "sup_seq":
        return
    
    # ====================
    # Memorisation profile
    # ====================
    mem_prof = (
        pn.ggplot(
            df.query("(time >= cohort) & (zero_not_in_cband == '*') & (cohort <= 95000) & (time <= 95000)"),
            pn.aes("time", "cohort", alpha="memorisation", colour="model_size")
        ) +
        pn.geom_tile(pn.aes(width=1000, height=1000)) +
        pn.facet_wrap("model_size", ncol=3) +
        pn.labs(title="", x="Checkpoint Step", y="Treatment Step") +
        pn.scale_x_continuous(
            expand=(0, 0), 
            breaks=range(10000, 98000, 10000), 
            labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 == 0 else "" for i, v in enumerate(x)],
        ) +
        pn.scale_y_continuous(
            expand=(0, 0), 
            breaks=range(10000, 98000, 10000), 
            labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 == 0 else "" for i, v in enumerate(x)],
        ) +
        pn.scale_alpha(range=(.5, 1.), guide=None) +
        pn.scale_colour_discrete(guide=None) +
        # pn.scale_colour_manual(guide=None, values=["black", "red", "green", "yellow", "orange", "blue"]) +
        pn.theme_bw(base_size=11) +
        pn.theme(panel_grid_major=pn.element_blank(), plot_margin=0, plot_background=None, figure_size=(8, 3.2))
    )

    # mem_prof.save(str(out_path / f"mem_prof_{metric}.svg"))
    pn.save_as_pdf_pages([mem_prof], str(out_path / f"mem_prof_{metric}.pdf"))
    
    # ======================
    # Learning rate schedule
    # ======================
    total_steps = df["time"].max()
    warmup_steps = np.ceil(0.01 * total_steps)
    max_lr = 0.001
    base_lr = 0.1 * max_lr

    def cosine_lr_with_warmup(
        current_step: int, warmup_steps: int, total_steps: int, base_lr: float, max_lr: float
    ) -> float:
        if current_step <= warmup_steps:
            return base_lr + (max_lr - base_lr) * current_step / warmup_steps
        cosine_steps = total_steps - warmup_steps
        current_step = current_step - warmup_steps
        return base_lr + 0.5 * (max_lr - base_lr) * (1 + np.cos(np.pi * current_step / cosine_steps))

    lr_df = pd.DataFrame(
        [(i, cosine_lr_with_warmup(i, warmup_steps, total_steps, base_lr, max_lr)) for i in range(0, 96000, 1000)],
        columns=["time", "lr"],
    )

    lr_plot = (
        pn.ggplot(lr_df, pn.aes("time", "lr"))
        +
        # pn.geom_line() +
        pn.geom_segment(pn.aes(y=0, yend="lr", xend="time"))
        + pn.geom_point(size=0.08)
        + pn.scale_x_continuous(
            # expand=(0, 0),
            breaks=range(0, 98000, 10000),
            labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 != 0 else "" for i, v in enumerate(x)],
        )
        + pn.scale_y_continuous(
            breaks=np.linspace(0, max_lr, 6),
            labels=lambda x: [rf"${round(i * 1000, 3)} \times 10^{{-3}}$" if i > 0 else "0" for i in x],
        )
        + pn.labs(y="", x="Checkpoint Timestep")
        +
        # pn.scale_y_log10() +
        pn.theme_bw(base_size=11)
        + pn.theme(plot_background=None, figure_size=(4, 2))
    )

    # lr_plot.save(str(out_path / "lr_plot.svg"))
    pn.save_as_pdf_pages([lr_plot], str(out_path / "lr_plot.pdf"))

    # =======================
    # Persistent memorisation
    # =======================
    event_df = pd.concat(
        [
            pd.read_parquet(p).assign(model_size=p.name.split("_")[1].split("-")[0].strip())
            for p in data_path.glob("cs*sup_seq-event_boot*")
        ]
    )

    event_df = (
        event_df.query("model_size != '2.8b'")
        .assign(
            model_size=lambda _df: as_cat(_df["model_size"].str.upper()),
            memorisation=lambda _df: -_df["ATT"],
            lower=lambda _df: -_df["lower"],
            upper=lambda _df: -_df["upper"],
            std_error=lambda _df: -_df["std_error"],
        )
        .assign(
            stdupper=lambda _df: _df["memorisation"] + (2 * _df["std_error"]),
            stdlower=lambda _df: _df["memorisation"] - (2 * _df["std_error"]),
        )
        .assign(zero_not_in_cf=lambda _df: ((_df["stdlower"] > 0) | (_df["stdupper"] < 0)).map({True: "*", False: ""}))
    )

    per_mem = (
        pn.ggplot(
            event_df.query(
                "(relative_period >= 0) & (relative_period <= 95000) & (zero_not_in_cband == '*') & (memorisation > 0)"
            ),
            pn.aes("relative_period", "memorisation", colour="model_size", ymax="upper", ymin="lower"),
        )
        + pn.geom_line()
        + pn.geom_point()
        +
        # pn.geom_pointrange() +
        pn.scale_x_sqrt(labels=lambda x: [f"{v / 1000:.0f}k" if v != 0 else "0" for v in x])
        + pn.scale_y_sqrt(labels=lambda x: [round(i, 3) if i > 0 else "0" for i in x])
        + pn.labs(x="Timesteps after treatment", y="", colour="")
        +
        # pn.scale_colour_cmap_d("brewer") +
        pn.theme_bw(base_size=11)
        + pn.theme(plot_background=None, legend_box_margin=0, legend_box_spacing=0.01, figure_size=(4, 2.6))
    )
    # per_mem.save(str(out_path / f"per_mem_{metric}.svg"))
    pn.save_as_pdf_pages([per_mem], str(out_path / f"per_mem_{metric}.pdf"))

    # ==========================
    # Instantaneous memorisation
    # ==========================
    ins_mem = (
        pn.ggplot(
            df.query("(time == cohort) & (zero_not_in_cband == '*') & (cohort <= 95000) & (time <= 95000)"),
            pn.aes("cohort", "memorisation", colour="model_size"),
        )
        + pn.geom_line(alpha=0.2)
        + pn.geom_point(alpha=0.2)
        + pn.geom_smooth(se=False)
        + pn.labs(title="", x="Checkpoint Step", y="", colour="")
        + pn.scale_x_continuous(
            # expand=(0, 0),
            breaks=range(10000, 98000, 10000),
            labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 == 0 else "" for i, v in enumerate(x)],
        )
        + pn.scale_y_sqrt(
            # expand=(0, 0),
            breaks=np.linspace(0.0, 0.1, 6),
            labels=lambda x: [round(i, 3) if i > 0 else "0" for i in x],
            # labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 == 0 else "" for i, v in enumerate(x)],
        )
        +
        # pn.scale_fill_gradient2(low="white", high="red", guide=None, midpoint=0) +
        pn.theme_bw(base_size=11)
        + pn.theme(plot_background=None, legend_box_margin=0, legend_box_spacing=0.01, figure_size=(4, 2.6))
    )

    # ins_mem.save(str(out_path / f"ins_mem_{metric}.svg"))
    pn.save_as_pdf_pages([ins_mem], str(out_path / f"ins_mem_{metric}.pdf"))


    # ==================
    # Final memorisation
    # ==================
    fin_mem = (
        pn.ggplot(
            (
                df.query("(cohort <= 95000) & (time == 95000) & (memorisation >= 0)").assign(
                    alpha=lambda _df: _df["zero_not_in_cband"].map({"": 0.3, "*": 1.0})
                )
            ),
            pn.aes("cohort", "memorisation", colour="model_size", alpha="alpha"),
        )
        + pn.geom_segment(pn.aes(y=0, yend="memorisation", xend="cohort"))
        + pn.geom_point()
        + pn.facet_wrap("model_size", ncol=3)
        + pn.coord_cartesian(ylim=(0, 0.04))
        + pn.labs(y="", x="Treatment Step", colour="", alpha="")
        + pn.scale_alpha_continuous(range=(0.28, 1.0), guide=None)
        + pn.scale_colour_discrete(guide=None)
        + pn.scale_x_continuous(
            breaks=range(10000, 98000, 10000),
            labels=lambda x: [f"{v / 1000:.0f}k" if i % 2 == 0 else "" for i, v in enumerate(x)],
        )
        + pn.scale_y_continuous(labels=lambda x: [round(i, 3) if i > 0 else "0" for i in x])
        + pn.theme_bw(base_size=11)
        + pn.theme(plot_margin=0.002, plot_background=None, figure_size=(8, 3.2))
    )

    # fin_mem.save(str(out_path / f"fin_mem_{metric}.svg"))
    pn.save_as_pdf_pages([fin_mem], str(out_path / f"fin_mem_{metric}.pdf"))

    # =========================
    # Correlations across sizes
    # =========================
    def compute_correlations(
        df: pd.DataFrame, method: Literal["pearson", "kendall", "spearman"] = "pearson"
    ) -> pd.DataFrame:
        return (
            df.set_index(["cohort", "time", "model_size"])["memorisation"]
            .unstack("model_size")
            .corr(method)
            .reset_index()
            .rename(columns={"model_size": "other"})
            .melt(id_vars="other")
            .assign(model_size=lambda _df: as_cat(_df["model_size"]), other=lambda _df: as_cat(_df["other"]))
            .sort_values(["model_size", "other"])
            .assign(pair=lambda _df: _df.apply(set, axis=1))
            .drop_duplicates(subset=["pair"])
            .drop(columns=["pair"])
        )

    def plot_corr(df: pd.DataFrame) -> pn.ggplot:
        return (
            pn.ggplot(df, pn.aes("other", "model_size", fill="value", label="round(value, 2)"))
            + pn.geom_tile(colour=None)
            + pn.geom_text(size=9)
            + pn.scale_x_discrete(expand=(0, 0))
            + pn.scale_y_discrete(expand=(0, 0))
            + pn.labs(x="", y="")
            + pn.scale_fill_gradient2(mid="white", high="darkgrey", midpoint=0, guide=None)
            + pn.theme_bw(base_size=11)
            + pn.theme(panel_grid=pn.element_blank(), figure_size=(4, 2.4))
        )

    pcorr = compute_correlations(df, "pearson")
    mem_prof_corr = plot_corr(pcorr)
    # mem_prof_corr.save(str(out_path / f"mem_prof_corr_{metric}.svg"))
    pn.save_as_pdf_pages([mem_prof_corr], str(out_path / f"mem_prof_corr_{metric}.pdf"))


if __name__ == "__main__":
    for metric in ("sup_seq", "acc_seq", "avg_rank", "entr_seq"):
        main(metric)
