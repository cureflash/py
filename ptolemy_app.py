from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import least_squares

MARS_PERIOD_DAYS = 686.98
EARTH_PERIOD_DAYS = 365.256


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    elements: tuple[str, ...]
    explanation: str


STAGES = (
    Stage(1, "従円のみ", ("従円",), "地球を中心に、惑星が等速円運動すると仮定します。"),
    Stage(2, "+ 周転円", ("従円", "周転円"), "従円上の点を中心に周転円を回し、逆行を表現します。"),
    Stage(3, "+ 離心円", ("従円", "周転円", "離心円"), "従円の中心を地球からずらし、見かけの角速度の非対称性を加えます。"),
    Stage(4, "+ エカント", ("従円", "周転円", "離心円", "エカント"), "エカントから見た角速度を等速とし、さらに非一様な見かけの運動を表現します。"),
)


def wrap_deg(angle_deg: np.ndarray | float) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=float)
    return (angle + 180.0) % 360.0 - 180.0


def ptolemy_xy(days: np.ndarray, params: np.ndarray, stage_number: int) -> tuple[np.ndarray, np.ndarray]:
    """Return geocentric x/y positions for the selected cumulative model stage."""
    days = np.asarray(days, dtype=float)

    deferent_phase = params[0]
    epicycle_ratio = 0.0
    epicycle_phase = 0.0
    eccentricity = 0.0
    eccentric_angle = 0.0

    if stage_number >= 2:
        epicycle_ratio = params[1]
        epicycle_phase = params[2]
    if stage_number >= 3:
        eccentricity = params[3]
        eccentric_angle = params[4]

    alpha = 2.0 * np.pi * days / MARS_PERIOD_DAYS + deferent_phase
    center = np.array(
        [
            eccentricity * np.cos(eccentric_angle),
            eccentricity * np.sin(eccentric_angle),
        ]
    )

    if stage_number >= 4:
        # Earth E = (0, 0), deferent center C = center, equant Q = 2C.
        equant = 2.0 * center
        ux = np.cos(alpha)
        uy = np.sin(alpha)

        # Intersect the uniformly rotating ray from Q with the unit deferent.
        dot = center[0] * ux + center[1] * uy
        discriminant = np.maximum(dot * dot - (eccentricity * eccentricity - 1.0), 0.0)
        distance = -dot + np.sqrt(discriminant)
        deferent_x = equant[0] + distance * ux
        deferent_y = equant[1] + distance * uy
    else:
        deferent_x = center[0] + np.cos(alpha)
        deferent_y = center[1] + np.sin(alpha)

    if stage_number >= 2:
        beta = 2.0 * np.pi * days / EARTH_PERIOD_DAYS + epicycle_phase
        planet_x = deferent_x + epicycle_ratio * np.cos(beta)
        planet_y = deferent_y + epicycle_ratio * np.sin(beta)
    else:
        planet_x = deferent_x
        planet_y = deferent_y

    return planet_x, planet_y


def ptolemy_longitude(days: np.ndarray, params: np.ndarray, stage_number: int) -> np.ndarray:
    x, y = ptolemy_xy(days, params, stage_number)
    return np.degrees(np.arctan2(y, x)) % 360.0


def _initial_guess(stage_number: int, rng: np.random.Generator) -> np.ndarray:
    deferent_phase = rng.uniform(-np.pi, np.pi)
    if stage_number == 1:
        return np.array([deferent_phase])

    epicycle_ratio = rng.uniform(0.4, 0.9)
    epicycle_phase = rng.uniform(-np.pi, np.pi)
    if stage_number == 2:
        return np.array([deferent_phase, epicycle_ratio, epicycle_phase])

    eccentricity = rng.uniform(0.02, 0.14)
    eccentric_angle = rng.uniform(-np.pi, np.pi)
    return np.array(
        [deferent_phase, epicycle_ratio, epicycle_phase, eccentricity, eccentric_angle]
    )


def _bounds(stage_number: int) -> tuple[np.ndarray, np.ndarray]:
    if stage_number == 1:
        return np.array([-np.pi]), np.array([np.pi])
    if stage_number == 2:
        return (
            np.array([-np.pi, 0.05, -np.pi]),
            np.array([np.pi, 1.50, np.pi]),
        )
    return (
        np.array([-np.pi, 0.05, -np.pi, 0.0, -np.pi]),
        np.array([np.pi, 1.50, np.pi, 0.25, np.pi]),
    )


def fit_model(days: np.ndarray, observed_longitude: np.ndarray, stage_number: int) -> dict:
    """Fit the selected model to angular observations using circular residuals."""
    rng = np.random.default_rng(4200 + stage_number)
    lower, upper = _bounds(stage_number)
    best = None

    for _ in range(16):
        x0 = _initial_guess(stage_number, rng)
        result = least_squares(
            lambda p: np.radians(
                wrap_deg(ptolemy_longitude(days, p, stage_number) - observed_longitude)
            ),
            x0,
            bounds=(lower, upper),
            max_nfev=5000,
        )
        prediction = ptolemy_longitude(days, result.x, stage_number)
        residual_deg = wrap_deg(prediction - observed_longitude)
        rms_deg = float(np.sqrt(np.mean(residual_deg**2)))

        if best is None or rms_deg < best["rms_deg"]:
            best = {
                "params": result.x,
                "prediction_deg": prediction,
                "residual_deg": residual_deg,
                "rms_deg": rms_deg,
            }

    assert best is not None
    return best


def _kepler_position(
    days: np.ndarray,
    period_days: float,
    eccentricity: float,
    semimajor_axis: float,
    mean_anomaly_0: float,
    perihelion_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean_anomaly = (mean_anomaly_0 + 2.0 * np.pi * days / period_days) % (2.0 * np.pi)
    eccentric_anomaly = mean_anomaly.copy()

    for _ in range(12):
        eccentric_anomaly -= (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        ) / (1.0 - eccentricity * np.cos(eccentric_anomaly))

    x_orbit = semimajor_axis * (np.cos(eccentric_anomaly) - eccentricity)
    y_orbit = semimajor_axis * np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly)

    c = np.cos(perihelion_longitude)
    s = np.sin(perihelion_longitude)
    x = c * x_orbit - s * y_orbit
    y = s * x_orbit + c * y_orbit
    return x, y


def make_demo_observations() -> pd.DataFrame:
    """Create a physically motivated demo target; this is not real observational data."""
    days = np.arange(0.0, 801.0, 5.0)
    earth_x, earth_y = _kepler_position(
        days,
        EARTH_PERIOD_DAYS,
        eccentricity=0.0167,
        semimajor_axis=1.0,
        mean_anomaly_0=1.0,
        perihelion_longitude=1.8,
    )
    mars_x, mars_y = _kepler_position(
        days,
        MARS_PERIOD_DAYS,
        eccentricity=0.0934,
        semimajor_axis=1.5237,
        mean_anomaly_0=4.0,
        perihelion_longitude=0.9,
    )
    longitude = np.degrees(np.arctan2(mars_y - earth_y, mars_x - earth_x)) % 360.0
    return pd.DataFrame({"day": days, "longitude_deg": longitude})


def normalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {column.lower().strip(): column for column in frame.columns}

    if "longitude_deg" not in columns:
        raise ValueError("CSVには longitude_deg 列が必要です。")

    longitude = pd.to_numeric(frame[columns["longitude_deg"]], errors="coerce")

    if "day" in columns:
        day = pd.to_numeric(frame[columns["day"]], errors="coerce")
    elif "date" in columns:
        date = pd.to_datetime(frame[columns["date"]], errors="coerce")
        day = (date - date.min()).dt.total_seconds() / 86400.0
    else:
        raise ValueError("CSVには day または date 列が必要です。")

    normalized = pd.DataFrame({"day": day, "longitude_deg": longitude}).dropna()
    normalized = normalized.sort_values("day").drop_duplicates(subset="day")

    if len(normalized) < 8:
        raise ValueError("フィットには8点以上の観測値が必要です。")

    normalized["day"] = normalized["day"] - normalized["day"].iloc[0]
    normalized["longitude_deg"] = normalized["longitude_deg"] % 360.0
    return normalized.reset_index(drop=True)


def stage_parameter_table(params: np.ndarray, stage_number: int) -> pd.DataFrame:
    rows = [
        ("従円初期位相", f"{np.degrees(params[0]):.2f}°"),
        ("従円周期", f"{MARS_PERIOD_DAYS:.2f} 日（固定）"),
    ]
    if stage_number >= 2:
        rows.extend(
            [
                ("周転円半径 / 従円半径", f"{params[1]:.4f}"),
                ("周転円初期位相", f"{np.degrees(params[2]):.2f}°"),
                ("周転円周期", f"{EARTH_PERIOD_DAYS:.3f} 日（固定）"),
            ]
        )
    if stage_number >= 3:
        rows.extend(
            [
                ("離心量 / 従円半径", f"{params[3]:.4f}"),
                ("離心方向", f"{np.degrees(params[4]):.2f}°"),
            ]
        )
    if stage_number >= 4:
        rows.append(("エカント位置", "従円中心をはさんで地球と対称"))
    return pd.DataFrame(rows, columns=["パラメータ", "値"])


def plot_longitude(data: pd.DataFrame, fit: dict):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(data["day"], data["longitude_deg"], s=18, label="観測 / 比較対象")
    ax.plot(data["day"], fit["prediction_deg"], linewidth=1.7, label="モデル予測")
    ax.set_xlabel("基準日からの日数")
    ax.set_ylabel("黄経 [deg]")
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 60))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residual(data: pd.DataFrame, fit: dict):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axhline(0.0, linewidth=1.0)
    ax.plot(data["day"], fit["residual_deg"], marker="o", markersize=3, linewidth=1.0)
    ax.set_xlabel("基準日からの日数")
    ax.set_ylabel("予測 − 観測 [deg]")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_geometry(days: np.ndarray, fit: dict, stage_number: int):
    params = fit["params"]
    sample_days = np.linspace(days.min(), days.max(), 900)
    px, py = ptolemy_xy(sample_days, params, stage_number)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.plot(px, py, linewidth=1.0, label="惑星の軌跡")
    ax.scatter([0.0], [0.0], s=55, marker="o", label="地球")

    theta = np.linspace(0, 2.0 * np.pi, 500)
    eccentricity = params[3] if stage_number >= 3 else 0.0
    eccentric_angle = params[4] if stage_number >= 3 else 0.0
    center = np.array(
        [eccentricity * np.cos(eccentric_angle), eccentricity * np.sin(eccentric_angle)]
    )
    ax.plot(
        center[0] + np.cos(theta),
        center[1] + np.sin(theta),
        linestyle="--",
        linewidth=1.0,
        label="従円",
    )

    if stage_number >= 3:
        ax.scatter([center[0]], [center[1]], s=45, marker="x", label="従円中心")
    if stage_number >= 4:
        equant = 2.0 * center
        ax.scatter([equant[0]], [equant[1]], s=45, marker="+", label="エカント")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x（従円半径 = 1）")
    ax.set_ylabel("y（従円半径 = 1）")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="段階式プトレマイオスモデル検証", layout="wide")
    st.title("段階式プトレマイオスモデル検証")
    st.caption("火星の見かけの黄経を、要素を一つずつ増やした天動説モデルで比較します。")

    if "stage" not in st.session_state:
        st.session_state.stage = 1

    stage_columns = st.columns(4)
    for index, stage in enumerate(STAGES):
        button_label = f"{stage.number}. {stage.name}"
        if stage_columns[index].button(
            button_label,
            use_container_width=True,
            type="primary" if st.session_state.stage == stage.number else "secondary",
            key=f"stage_{stage.number}",
        ):
            st.session_state.stage = stage.number
            st.rerun()

    selected = STAGES[st.session_state.stage - 1]
    st.subheader(f"段階 {selected.number}: {selected.name}")
    st.write(selected.explanation)
    st.caption("使用中: " + " / ".join(selected.elements))

    uploaded = st.file_uploader(
        "観測CSV（任意）",
        type=["csv"],
        help="day,longitude_deg または date,longitude_deg の2列を使います。",
    )

    if uploaded is None:
        observations = make_demo_observations()
        st.info("現在はKepler楕円軌道から作った比較用の模擬観測を使用しています。実観測データではありません。")
    else:
        try:
            observations = normalize_observations(pd.read_csv(uploaded))
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    days = observations["day"].to_numpy(dtype=float)
    longitude = observations["longitude_deg"].to_numpy(dtype=float)

    with st.spinner("モデルを観測値へフィットしています…"):
        selected_fit = fit_model(days, longitude, selected.number)
        all_fits = [fit_model(days, longitude, stage.number) for stage in STAGES]

    metric_left, metric_right = st.columns(2)
    metric_left.metric("RMS残差", f"{selected_fit['rms_deg']:.3f}°")
    metric_right.metric(
        "最大絶対残差",
        f"{np.max(np.abs(selected_fit['residual_deg'])):.3f}°",
    )

    plot_left, plot_right = st.columns([1.4, 1.0])
    with plot_left:
        st.pyplot(plot_longitude(observations, selected_fit), clear_figure=True)
        st.pyplot(plot_residual(observations, selected_fit), clear_figure=True)
    with plot_right:
        st.pyplot(plot_geometry(days, selected_fit, selected.number), clear_figure=True)
        st.dataframe(
            stage_parameter_table(selected_fit["params"], selected.number),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("段階ごとの比較")
    comparison = pd.DataFrame(
        {
            "段階": [f"{stage.number}. {stage.name}" for stage in STAGES],
            "使用要素": [" / ".join(stage.elements) for stage in STAGES],
            "RMS残差 [deg]": [fit["rms_deg"] for fit in all_fits],
            "最大絶対残差 [deg]": [
                float(np.max(np.abs(fit["residual_deg"]))) for fit in all_fits
            ],
        }
    )
    st.dataframe(
        comparison.style.format(
            {"RMS残差 [deg]": "{:.3f}", "最大絶対残差 [deg]": "{:.3f}"}
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "この実装は教育・検証用の簡略化モデルです。従円周期を火星の平均公転周期、周転円周期を1年に固定し、各段階で残りのパラメータを観測値に最小二乗フィットします。"
    )


if __name__ == "__main__":
    main()
