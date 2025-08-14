# executive_summary_agent.py
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from dotenv import load_dotenv

# LLM backends (OpenAI default; simple switcher for Mixtral/Qwen via env/config)
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

SUPPORTED_MODELS = {
    "openai:gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "openai:gpt-4o": ("openai", "gpt-4o"),
    "openai:gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "mixtral:8x7b": ("other", "mixtral-8x7b"),
    "qwen:2.5": ("other", "qwen-2.5"),
}

# Date split constant
SPLIT_DATE = pd.to_datetime("2016-05-23")

load_dotenv()

@dataclass
class PriceChange:
    store: str
    date: datetime
    old_price: float
    new_price: float
    pct_change: float


class ExecutiveSummaryAgent:
    """
    Generates executive summaries using both historical (before SPLIT_DATE) 
    and forecasted (after SPLIT_DATE) data for forward-looking recommendations.
    """

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        temperature: float = 0.5,
        calendar_path: str = "full_updated_calendar.csv",
        sell_prices_path: str = "sell_prices.csv"
    ):
        provider, actual_model = SUPPORTED_MODELS.get(model_name, ("openai", "gpt-4o-mini"))

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.llm = ChatOpenAI(model_name=actual_model, temperature=temperature, openai_api_key=api_key)
        else:
            raise ValueError(f"Model provider for '{model_name}' not wired yet.")

        # Load CSVs
        self.calendar_df = self._safe_read_csv(calendar_path)
        self.sell_prices_df = self._safe_read_csv(sell_prices_path)

        if self.calendar_df is not None and "date" in self.calendar_df.columns:
            try:
                self.calendar_df["date"] = pd.to_datetime(self.calendar_df["date"], errors="coerce")
            except Exception:
                pass

    def generate_report(
        self,
        parsed_query: Dict,
        forecast_data_dict: Dict[str, pd.DataFrame],
        metrics: List[Dict[str, Any]],
        vrp_result: Optional[Tuple],
        data_loader
    ) -> str:
        category = parsed_query.get("category")
        item = parsed_query.get("item")
        stores = parsed_query.get("stores", [])
        date_range = parsed_query.get("date_range")
        start_dt, end_dt = self._parse_date_range(date_range)

        # Historical patterns (< SPLIT_DATE)
        hist_price_changes = self._detect_price_changes(category, item, stores, end_dt=SPLIT_DATE - pd.Timedelta(days=1))
        hist_events = self._collect_events_and_snap(data_loader, end_dt=SPLIT_DATE - pd.Timedelta(days=1))
        hist_risks = self._detect_risks(parsed_query, forecast_data_dict, metrics, {}, end_dt=SPLIT_DATE - pd.Timedelta(days=1), use_forecast=False)

        # Future projections (≥ SPLIT_DATE)
        lead_times = {}
        route_sequence = []
        total_distance = total_time_hours = total_cost = None
        if vrp_result:
            if len(vrp_result) >= 6:
                _, route_order, total_distance, total_time, total_cost, lead_times = vrp_result
                total_time_hours = total_time / 3600 if total_time else None
            else:
                _, route_order, total_distance, total_time, total_cost = vrp_result
                total_time_hours = total_time / 3600 if total_time else None
                lead_times = {}
            if route_order is not None:
                for idx in route_order:
                    if idx == 0:
                        route_sequence.append("Distribution Center")
                    else:
                        pos = idx - 1
                        if 0 <= pos < len(stores):
                            route_sequence.append(stores[pos])

        future_price_changes = self._detect_price_changes(category, item, stores, start_dt=SPLIT_DATE)
        future_events = self._collect_events_and_snap(data_loader, start_dt=SPLIT_DATE)
        future_risks = self._detect_risks(parsed_query, forecast_data_dict, metrics, lead_times, start_dt=SPLIT_DATE, use_forecast=True)

        # Build context
        llm_context = {
            "overview": {
                "category": category,
                "item": item,
                "stores": ", ".join(sorted(stores)) if stores else "—",
                "date_range": f"{start_dt.date().isoformat()} to {end_dt.date().isoformat()}" if start_dt and end_dt else "unspecified"
            },
            "historical": {
                "price_changes": self._humanize_price_changes(hist_price_changes),
                "events": hist_events,
                "risks": hist_risks
            },
            "future": {
                "price_changes": self._humanize_price_changes(future_price_changes),
                "events": future_events,
                "risks": future_risks
            },
            "logistics_notes": self._logistics_notes(route_sequence, total_distance, total_time_hours, total_cost)
        }

        # LLM prompt
        prompt = self._summary_prompt().format(**llm_context)
        response = self.llm.invoke(prompt)
        return response.content

    def _detect_price_changes(
        self,
        category: str,
        item: str,
        stores: List[str],
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        min_pct_threshold: float = 0.05
    ) -> List[PriceChange]:
        results: List[PriceChange] = []
        if self.sell_prices_df is None:
            return results

        df = self.sell_prices_df.copy()
        for store in stores:
            item_id = f"{category}_{item}"
            store_id = store
            sub = df[(df["item_id"] == item_id) & (df["store_id"] == store_id)].copy()
            if sub.empty:
                continue

            if self.calendar_df is not None and "wm_yr_wk" in self.calendar_df.columns:
                cal = self.calendar_df[["wm_yr_wk", "date"]].drop_duplicates().copy()
                cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
                sub = sub.merge(cal, on="wm_yr_wk", how="left")
            else:
                sub["date"] = pd.NaT

            sub.sort_values("date", inplace=True)
            if start_dt:
                sub = sub[sub["date"] >= start_dt]
            if end_dt:
                sub = sub[sub["date"] <= end_dt]

            prices = sub["sell_price"].astype(float).values
            dates = sub["date"].values
            for i in range(1, len(prices)):
                old_p, new_p = prices[i - 1], prices[i]
                if old_p <= 0:
                    continue
                pct = (new_p - old_p) / old_p
                if abs(pct) >= min_pct_threshold:
                    date_val = dates[i] if not pd.isna(dates[i]) else None
                    results.append(PriceChange(
                        store=store,
                        date=pd.to_datetime(date_val) if date_val is not None else None,
                        old_price=float(old_p),
                        new_price=float(new_p),
                        pct_change=float(pct)
                    ))
        return results

    def _collect_events_and_snap(self, data_loader, start_dt: Optional[datetime] = None, end_dt: Optional[datetime] = None) -> List[str]:
        out: List[str] = []
        cal = getattr(data_loader, "calendar_data", None)
        if cal is None or "date" not in cal.columns:
            return out
        cal = cal.copy()
        if start_dt:
            cal = cal[cal["date"] >= start_dt]
        if end_dt:
            cal = cal[cal["date"] <= end_dt]
        for _, row in cal.iterrows():
            d = row.get("date")
            if pd.isna(d):
                continue
            d_str = pd.to_datetime(d).date().isoformat()
            if pd.notna(row.get("event_name_1")):
                out.append(f"{d_str}: {row.get('event_name_1')} ({row.get('event_type_1')})")
            if pd.notna(row.get("event_name_2")):
                out.append(f"{d_str}: {row.get('event_name_2')} ({row.get('event_type_2')})")
            for snap_col in ["snap_CA", "snap_TX", "snap_WI"]:
                if snap_col in cal.columns and row.get(snap_col) == 1:
                    out.append(f"{d_str}: SNAP disbursement ({snap_col.upper()})")
        return out

    def _detect_risks(
        self,
        parsed_query: Dict,
        forecast_data_dict: Dict[str, pd.DataFrame],
        metrics: List[Dict[str, Any]],
        lead_times: Dict[str, float],
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
        use_forecast: bool = True
    ) -> Dict[str, Any]:
        risk = {"stockout": [], "oversupply": [], "logistics": [], "notable_dates": []}
        metrics_by_store = {}
        for row in metrics or []:
            sku_id = row.get("SKU_ID", "")
            parts = sku_id.split("_")
            if len(parts) >= 5:
                store = f"{parts[-2]}_{parts[-1]}"
            elif len(parts) >= 4:
                store = parts[-1]
            else:
                continue
            metrics_by_store[store] = {
                "EOQ": float(row.get("EOQ", 0.0)),
                "ROP": float(row.get("ReorderPoint", 0.0)),
                "SS": float(row.get("SafetyStock", 0.0)),
            }

        for store, df in (forecast_data_dict or {}).items():
            avg_forecast = None
            if use_forecast:
                day_cols = [c for c in df.columns if c.isdigit()]
                if day_cols:
                    values = pd.to_numeric(df[day_cols].iloc[0], errors="coerce").fillna(0.0)
                    avg_forecast = float(values.mean())

            m = metrics_by_store.get(store, {})
            eoq, rop, ss = m.get("EOQ", 0.0), m.get("ROP", 0.0), m.get("SS", 0.0)
            lt_days = float(lead_times.get(store, 0.0)) / 24.0 if lead_times else 0.0

            if lt_days > 0 and (rop > 0 or ss > 0):
                if avg_forecast and (rop <= avg_forecast * lt_days * 1.1) and ss <= max(0.5, avg_forecast * 0.3):
                    risk["stockout"].append(f"{store}: buffer thin vs lead time; advance replenishment.")
            if avg_forecast and eoq > avg_forecast * 7.0:
                risk["oversupply"].append(f"{store}: EOQ high vs weekly demand; risk of overstock.")
        if lead_times:
            slow = [s for s, h in lead_times.items() if h and h > 24]
            if slow:
                risk["logistics"].append(f"Long lead times to {', '.join(slow)} may affect spikes.")
        if start_dt and end_dt:
            mid = start_dt + (end_dt - start_dt) / 2
            risk["notable_dates"].append(f"Mid-window around {mid.date().isoformat()} may require stock checks.")
        return risk

    def _humanize_price_changes(self, price_changes: List[PriceChange]) -> List[str]:
        out = []
        for pc in price_changes:
            when = pc.date.date().isoformat() if pc.date else "unspecified"
            direction = "increase" if pc.pct_change > 0 else "decrease"
            out.append(f"{pc.store}: {direction} of ~{abs(pc.pct_change)*100:.1f}% around {when}")
        return out

    def _logistics_notes(self, route_sequence, total_distance, total_time_hours, total_cost) -> List[str]:
        notes = []
        if route_sequence:
            notes.append(f"Route sequence: {' → '.join(route_sequence)}")
        if total_time_hours and total_time_hours > 48:
            notes.append("Multi-day travel time; consider staging/splitting loads.")
        if total_distance and total_distance > 2000:
            notes.append("Long route; sensitive to fuel/driver hours.")
        if total_cost and total_cost > 0:
            notes.append("Delivery cost significant; consolidate loads.")
        return notes

    def _summary_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["overview", "historical", "future", "logistics_notes"],
            template="""
You are an Executive Supply Chain Analyst. 
Use both historical (< 2016-05-23) and forecasted (≥ 2016-05-23) data. 
Blend past trends with future forecasts to create a concise, actionable **executive summary**.

Overview: {overview}

Historical patterns:
- Price changes: {historical[price_changes]}
- Events: {historical[events]}
- Risks: {historical[risks]}

Future outlook:
- Price changes: {future[price_changes]}
- Events: {future[events]}
- Risks: {future[risks]}

Logistics: {logistics_notes}

Guidelines:
- Focus on **what will happen, when, and what to do**.
- Highlight if historical patterns suggest a repeat in the future.
- Make 3–6 short bullet recommendations.
- Avoid repeating raw numbers already shown elsewhere.
"""
        )

    def _safe_read_csv(self, path: str) -> Optional[pd.DataFrame]:
        try:
            if os.path.exists(path):
                return pd.read_csv(path)
        except Exception:
            return None
        return None

    def _parse_date_range(self, dr: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
        if not dr or " to " not in str(dr):
            return (None, None)
        s, e = dr.split(" to ")
        try:
            return (pd.to_datetime(s, errors="coerce"), pd.to_datetime(e, errors="coerce"))
        except Exception:
            return (None, None)
