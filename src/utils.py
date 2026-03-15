from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from meteostat import Point, config, daily, hourly, stations
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.constants import (
    DEFAULT_MIN_HOURLY_OBSERVATIONS,
    FIGURES_DIR,
    MODEL_DATASET_FILENAME,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DAILY_FILENAME,
    RAW_DATA_DIR,
    RAW_HOURLY_FILENAME,
    TABLES_DIR,
    TEST_DAYS,
    VALIDATION_DAYS,
    VOLGOGRAD_ELEVATION,
    VOLGOGRAD_LATITUDE,
    VOLGOGRAD_LONGITUDE,
    VOLGOGRAD_NAME,
)

SPARSE_OPTIONAL_COLUMNS = {"wpgt_max", "tsun_sum", "rhum_mean", "dwpt_mean"}
CORE_REQUIRED_COLUMNS = {
    "target_tavg",
    "tmin",
    "tmax",
    "temp_range",
    "prcp_sum",
    "snow",
    "pres_mean",
    "wspd_mean",
}


def ensure_project_directories() -> None:
    """Создаёт рабочие папки проекта, если они ещё не существуют."""

    for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_meteostat_runtime() -> None:
    """Настраивает Meteostat для локального проекта."""

    ensure_project_directories()

    meteostat_root = RAW_DATA_DIR / "meteostat_cache"
    meteostat_cache = meteostat_root / "cache"
    stations_db_file = meteostat_root / "stations.db"

    meteostat_cache.mkdir(parents=True, exist_ok=True)

    config.cache_directory = str(meteostat_cache)
    config.stations_db_file = str(stations_db_file)
    config.block_large_requests = False


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Вычисляет расстояние между двумя точками на Земле в километрах."""

    radius_km = 6371.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return radius_km * c


def _strip_timezone(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Удаляет временную зону из индекса, если она присутствует."""

    if index.tz is None:
        return index
    return index.tz_convert(None)


def normalize_meteostat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Преобразует индекс Meteostat в обычный столбец даты."""

    normalized = frame.copy()
    normalized.index = _strip_timezone(pd.DatetimeIndex(pd.to_datetime(normalized.index)))
    normalized = normalized.reset_index().rename(columns={normalized.index.name or "index": "time"})
    normalized["time"] = pd.to_datetime(normalized["time"])
    normalized["date"] = normalized["time"].dt.normalize()
    return normalized


def get_nearby_station_candidates(
    latitude: float = VOLGOGRAD_LATITUDE,
    longitude: float = VOLGOGRAD_LONGITUDE,
    elevation: float = VOLGOGRAD_ELEVATION,
    limit: int = 8,
) -> pd.DataFrame:
    """Возвращает ближайшие к Волгограду станции Meteostat."""

    point = Point(latitude, longitude, elevation)
    station_candidates = stations.nearby(point, limit=limit)
    if station_candidates is None or station_candidates.empty:
        raise RuntimeError("Meteostat не вернул ближайшие станции для выбранных координат.")

    station_candidates = station_candidates.reset_index().rename(columns={"id": "station_id"})
    station_candidates["distance_km"] = station_candidates.apply(
        lambda row: haversine_distance_km(
            latitude,
            longitude,
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )
    if "distance" in station_candidates.columns:
        station_candidates["distance_km"] = (
            station_candidates["distance"].astype(float).div(1000).round(2)
        )

    return station_candidates.sort_values(["distance_km", "name"]).reset_index(drop=True)


def fetch_meteostat_data(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    latitude: float = VOLGOGRAD_LATITUDE,
    longitude: float = VOLGOGRAD_LONGITUDE,
    elevation: float = VOLGOGRAD_ELEVATION,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Загружает данные Meteostat и при необходимости переходит на Point fallback."""

    ensure_project_directories()
    configure_meteostat_runtime()

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    hourly_end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(hours=1)

    candidates = get_nearby_station_candidates(
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
    )

    errors: list[str] = []

    for row in candidates.head(5).itertuples(index=False):
        try:
            station_id = getattr(row, "station_id")
            hourly_data = hourly(station_id, start_ts, hourly_end_ts).fetch()
            daily_data = daily(station_id, start_ts, end_ts).fetch()

            if hourly_data is None or daily_data is None:
                errors.append(f"Станция {station_id}: Meteostat вернул пустой объект.")
                continue

            if hourly_data.empty and daily_data.empty:
                errors.append(f"Станция {station_id}: пустой ответ.")
                continue

            selection = {
                "source_type": "station",
                "station_id": station_id,
                "station_name": getattr(row, "name"),
                "country": getattr(row, "country"),
                "region": getattr(row, "region"),
                "distance_km": float(getattr(row, "distance_km")),
                "latitude": float(getattr(row, "latitude")),
                "longitude": float(getattr(row, "longitude")),
                "elevation": float(getattr(row, "elevation")) if pd.notna(getattr(row, "elevation")) else np.nan,
            }
            return selection, candidates, daily_data, hourly_data
        except Exception as exc:  # pragma: no cover - сеть и внешние данные
            errors.append(f"Станция {getattr(row, 'station_id')}: {exc}")

    try:
        point = Point(latitude, longitude, elevation)
        hourly_data = hourly(point, start_ts, hourly_end_ts).fetch()
        daily_data = daily(point, start_ts, end_ts).fetch()

        if hourly_data is None or daily_data is None:
            raise RuntimeError("Point fallback вернул пустой объект.")

        if hourly_data.empty and daily_data.empty:
            raise RuntimeError("Point fallback вернул пустые данные.")

        selection = {
            "source_type": "point",
            "station_id": "POINT",
            "station_name": VOLGOGRAD_NAME,
            "country": "RU",
            "region": "Волгоградская область",
            "distance_km": 0.0,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
        }
        return selection, candidates, daily_data, hourly_data
    except Exception as exc:  # pragma: no cover - сеть и внешние данные
        joined_errors = "; ".join(errors) if errors else "Подходящие станции не найдены."
        raise RuntimeError(
            "Не удалось загрузить данные Meteostat ни по станции, ни по координатам. "
            f"Подробности: {joined_errors}; Point fallback: {exc}"
        ) from exc


def save_raw_snapshots(hourly: pd.DataFrame, daily: pd.DataFrame) -> tuple[Path, Path]:
    """Сохраняет сырые выгрузки Meteostat в папку data/raw."""

    ensure_project_directories()

    hourly_path = RAW_DATA_DIR / RAW_HOURLY_FILENAME
    daily_path = RAW_DATA_DIR / RAW_DAILY_FILENAME

    normalize_meteostat_frame(hourly).to_csv(hourly_path, index=False)
    normalize_meteostat_frame(daily).to_csv(daily_path, index=False)
    return hourly_path, daily_path


def load_raw_snapshots() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает ранее сохранённые сырые данные из data/raw."""

    hourly_path = RAW_DATA_DIR / RAW_HOURLY_FILENAME
    daily_path = RAW_DATA_DIR / RAW_DAILY_FILENAME

    if not hourly_path.exists() or not daily_path.exists():
        raise FileNotFoundError("Локальные raw-снимки Meteostat не найдены.")

    hourly_frame = pd.read_csv(hourly_path, parse_dates=["time", "date"]).set_index("time")
    daily_frame = pd.read_csv(daily_path, parse_dates=["time", "date"]).set_index("time")
    return daily_frame, hourly_frame


def aggregate_hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует почасовые данные до уровня суток."""

    hourly_norm = normalize_meteostat_frame(hourly)

    snow_column = "snow" if "snow" in hourly_norm.columns else "snwd" if "snwd" in hourly_norm.columns else None

    aggregation_plan: dict[str, tuple[str, str]] = {
        "target_tavg": ("temp", "mean"),
        "tmin": ("temp", "min"),
        "tmax": ("temp", "max"),
        "prcp_sum": ("prcp", "sum"),
        "pres_mean": ("pres", "mean"),
        "wspd_mean": ("wspd", "mean"),
        "wpgt_max": ("wpgt", "max"),
        "rhum_mean": ("rhum", "mean"),
        "dwpt_mean": ("dwpt", "mean"),
        "tsun_sum": ("tsun", "sum"),
    }
    if snow_column is not None:
        aggregation_plan["snow"] = (snow_column, "max")

    named_aggregations = {
        output_name: pd.NamedAgg(column=source_column, aggfunc=agg_func)
        for output_name, (source_column, agg_func) in aggregation_plan.items()
        if source_column in hourly_norm.columns
    }

    if "temp" in hourly_norm.columns:
        named_aggregations["temp_observations"] = pd.NamedAgg(column="temp", aggfunc="count")

    if not named_aggregations:
        raise RuntimeError("В почасовых данных отсутствуют нужные метеорологические столбцы.")

    aggregated = (
        hourly_norm.groupby("date")
        .agg(**named_aggregations)
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if {"tmin", "tmax"}.issubset(aggregated.columns):
        aggregated["temp_range"] = aggregated["tmax"] - aggregated["tmin"]

    return aggregated


def prepare_daily_fallback(daily: pd.DataFrame) -> pd.DataFrame:
    """Подготавливает суточные данные как запасной источник признаков."""

    daily_norm = normalize_meteostat_frame(daily)

    rename_map = {
        "temp": "daily_tavg",
        "tavg": "daily_tavg",
        "tmin": "daily_tmin",
        "tmax": "daily_tmax",
        "prcp": "daily_prcp",
        "snwd": "daily_snow",
        "snow": "daily_snow",
        "pres": "daily_pres",
        "wspd": "daily_wspd",
        "wpgt": "daily_wpgt",
        "rhum": "daily_rhum",
        "tsun": "daily_tsun",
    }

    columns_to_keep = ["date"] + [column for column in rename_map if column in daily_norm.columns]
    fallback = daily_norm[columns_to_keep].rename(columns=rename_map)
    return fallback.sort_values("date").reset_index(drop=True)


def combine_hourly_and_daily_features(
    hourly_daily: pd.DataFrame,
    daily_fallback: pd.DataFrame,
    min_hourly_observations: int = DEFAULT_MIN_HOURLY_OBSERVATIONS,
) -> pd.DataFrame:
    """Объединяет агрегированные hourly-признаки и fallback из Daily."""

    dataset = hourly_daily.merge(daily_fallback, on="date", how="outer").sort_values("date").reset_index(drop=True)

    if "temp_observations" in dataset.columns:
        insufficient_obs = dataset["temp_observations"] < min_hourly_observations
        for column in ["target_tavg", "tmin", "tmax"]:
            if column in dataset.columns:
                dataset.loc[insufficient_obs, column] = np.nan

    fallback_pairs = {
        "target_tavg": "daily_tavg",
        "tmin": "daily_tmin",
        "tmax": "daily_tmax",
        "prcp_sum": "daily_prcp",
        "snow": "daily_snow",
        "pres_mean": "daily_pres",
        "wspd_mean": "daily_wspd",
        "wpgt_max": "daily_wpgt",
        "rhum_mean": "daily_rhum",
        "tsun_sum": "daily_tsun",
    }

    for main_column, fallback_column in fallback_pairs.items():
        if main_column in dataset.columns and fallback_column in dataset.columns:
            dataset[main_column] = dataset[main_column].combine_first(dataset[fallback_column])
        elif main_column not in dataset.columns and fallback_column in dataset.columns:
            dataset[main_column] = dataset[fallback_column]

    if {"tmin", "tmax"}.issubset(dataset.columns):
        dataset["temp_range"] = dataset["tmax"] - dataset["tmin"]

    daily_columns = [column for column in dataset.columns if column.startswith("daily_")]
    return dataset.drop(columns=daily_columns, errors="ignore")


def season_from_month(month: int) -> str:
    """Возвращает название сезона по номеру месяца."""

    if month in [12, 1, 2]:
        return "зима"
    if month in [3, 4, 5]:
        return "весна"
    if month in [6, 7, 8]:
        return "лето"
    return "осень"


def season_to_code(season: str) -> int:
    """Преобразует текстовый сезон в числовой код."""

    mapping = {"зима": 0, "весна": 1, "лето": 2, "осень": 3}
    return mapping[season]


def add_calendar_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Добавляет календарные и циклические признаки."""

    enriched = dataset.copy()
    enriched["month"] = enriched["date"].dt.month
    enriched["dayofyear"] = enriched["date"].dt.dayofyear
    enriched["dayofweek"] = enriched["date"].dt.dayofweek
    enriched["season"] = enriched["month"].apply(season_from_month)
    enriched["season_code"] = enriched["season"].apply(season_to_code)
    enriched["doy_sin"] = np.sin(2 * np.pi * enriched["dayofyear"] / 365.25)
    enriched["doy_cos"] = np.cos(2 * np.pi * enriched["dayofyear"] / 365.25)
    return enriched


def add_target_history_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Добавляет лаговые и скользящие признаки по температуре."""

    enriched = dataset.copy()
    history = enriched["target_tavg"].shift(1)

    for lag in [1, 2, 3, 7, 14, 30]:
        enriched[f"lag_{lag}"] = enriched["target_tavg"].shift(lag)

    enriched["rolling_mean_3"] = history.rolling(window=3).mean()
    enriched["rolling_mean_7"] = history.rolling(window=7).mean()
    enriched["rolling_mean_14"] = history.rolling(window=14).mean()
    enriched["rolling_std_7"] = history.rolling(window=7).std()
    enriched["rolling_std_14"] = history.rolling(window=14).std()

    return enriched


def apply_missing_value_strategy(dataset: pd.DataFrame) -> pd.DataFrame:
    """Аккуратно обрабатывает пропуски без использования информации из будущего."""

    cleaned = dataset.copy().sort_values("date").reset_index(drop=True)

    for column in ["prcp_sum", "snow", "tsun_sum"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(0.0)

    one_sided_fill_columns = [
        "tmin",
        "tmax",
        "temp_range",
        "prcp_sum",
        "snow",
        "pres_mean",
        "wspd_mean",
        "wpgt_max",
        "rhum_mean",
        "dwpt_mean",
        "tsun_sum",
    ]

    for column in one_sided_fill_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].ffill(limit=7)

    cleaned = cleaned.dropna(subset=["target_tavg"]).reset_index(drop=True)

    for column in list(cleaned.columns):
        if column in SPARSE_OPTIONAL_COLUMNS and column in cleaned.columns:
            missing_share = float(cleaned[column].isna().mean())
            if missing_share >= 0.95:
                cleaned = cleaned.drop(columns=column)
                continue

        if column in cleaned.columns and column not in {"date", "season", "target_tavg"} and cleaned[column].isna().all():
            cleaned = cleaned.drop(columns=column)

    cleaned = add_calendar_features(cleaned)
    cleaned = add_target_history_features(cleaned)

    required_columns = sorted(CORE_REQUIRED_COLUMNS.intersection(cleaned.columns)) + [
        "month",
        "dayofyear",
        "dayofweek",
        "season",
        "season_code",
        "doy_sin",
        "doy_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "rolling_std_14",
    ]
    required_columns = [column for column in required_columns if column in cleaned.columns]

    cleaned = cleaned.dropna(subset=required_columns).reset_index(drop=True)

    preferred_order = [
        "date",
        "target_tavg",
        "tmin",
        "tmax",
        "temp_range",
        "prcp_sum",
        "snow",
        "pres_mean",
        "wspd_mean",
        "wpgt_max",
        "rhum_mean",
        "dwpt_mean",
        "tsun_sum",
        "month",
        "dayofyear",
        "dayofweek",
        "season",
        "season_code",
        "doy_sin",
        "doy_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "rolling_std_14",
    ]

    ordered_columns = [column for column in preferred_order if column in cleaned.columns]
    return cleaned[ordered_columns]


def build_modeling_dataset(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Полный пайплайн построения датасета для моделирования."""

    selection, candidates, daily, hourly = fetch_meteostat_data(start=start, end=end)
    save_raw_snapshots(hourly=hourly, daily=daily)

    hourly_daily = aggregate_hourly_to_daily(hourly)
    daily_fallback = prepare_daily_fallback(daily)
    combined = combine_hourly_and_daily_features(hourly_daily=hourly_daily, daily_fallback=daily_fallback)
    dataset = apply_missing_value_strategy(combined)

    return dataset, selection, candidates


def get_or_build_modeling_dataset(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Строит датасет из Meteostat, а при неудаче использует локальные raw-снимки."""

    try:
        return build_modeling_dataset(start=start, end=end)
    except Exception as exc:
        daily_frame, hourly_frame = load_raw_snapshots()
        hourly_daily = aggregate_hourly_to_daily(hourly_frame)
        daily_fallback = prepare_daily_fallback(daily_frame)
        combined = combine_hourly_and_daily_features(hourly_daily=hourly_daily, daily_fallback=daily_fallback)
        dataset = apply_missing_value_strategy(combined)

        selection = {
            "source_type": "local_raw_csv",
            "station_id": "LOCAL_CACHE",
            "station_name": VOLGOGRAD_NAME,
            "country": "RU",
            "region": "Волгоградская область",
            "distance_km": 0.0,
            "latitude": VOLGOGRAD_LATITUDE,
            "longitude": VOLGOGRAD_LONGITUDE,
            "elevation": VOLGOGRAD_ELEVATION,
            "note": f"Использованы локальные raw-данные из data/raw из-за ошибки сетевой загрузки: {exc}",
        }
        candidates = pd.DataFrame()
        return dataset, selection, candidates


def save_processed_dataset(dataset: pd.DataFrame) -> Path:
    """Сохраняет итоговый датасет в папку data/processed."""

    ensure_project_directories()
    output_path = PROCESSED_DATA_DIR / MODEL_DATASET_FILENAME
    dataset.to_csv(output_path, index=False)
    return output_path


def describe_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:
    """Возвращает таблицу пропусков по столбцам."""

    summary = pd.DataFrame(
        {
            "missing_count": dataset.isna().sum(),
            "missing_share": dataset.isna().mean(),
        }
    )
    summary = summary[summary["missing_count"] > 0].sort_values(["missing_count", "missing_share"], ascending=False)
    return summary


def split_train_validation_test(
    dataset: pd.DataFrame,
    validation_days: int = VALIDATION_DAYS,
    test_days: int = TEST_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делит датасет на train, validation и test по времени."""

    if len(dataset) <= validation_days + test_days:
        raise ValueError("Недостаточно наблюдений для выделения validation и test интервалов.")

    train = dataset.iloc[: -(validation_days + test_days)].copy()
    validation = dataset.iloc[-(validation_days + test_days) : -test_days].copy()
    test = dataset.iloc[-test_days:].copy()

    return train, validation, test


def get_model_feature_columns(dataset: pd.DataFrame, exclude_columns: Iterable[str] | None = None) -> list[str]:
    """Возвращает список числовых признаков для моделей."""

    exclude = {"date", "season", "target_tavg"}
    if exclude_columns is not None:
        exclude.update(exclude_columns)

    feature_columns = [
        column
        for column in dataset.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(dataset[column])
    ]
    return sorted(feature_columns)


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series | np.ndarray, model_name: str) -> dict:
    """Вычисляет основные метрики регрессии."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)

    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "bias": float(np.mean(y_pred_array - y_true_array)),
    }
