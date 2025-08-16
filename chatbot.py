import os
import re
import json
import uuid
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from summary_agent import ExecutiveSummaryAgent
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

from data_loader import DataLoader
from checking import run_vrp_from_forecast
from streamlit_folium import st_folium

# LangChain imports (parsing only)
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser
import joblib
# Load environment variables
load_dotenv()


class ForecastingChatbot:
    """RAG-based chatbot for handling forecasting queries with context awareness"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.chat_history = []

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. Make sure it's defined in your .env file."
            )

        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        self.vector_store = None
        self.setup_vector_store()
        try:
            self.risk_model = joblib.load("risk_model.pkl")
        except Exception:
            self.risk_model = None

    def setup_vector_store(self):
        """Initialize FAISS vector store with product, calendar, and range data"""
        try:
            documents = []
            if self.data_loader.forecast_data_val is not None:
                documents.extend(self.data_loader.forecast_data_val['id'].tolist())
            if self.data_loader.forecast_data_eval is not None:
                documents.extend(self.data_loader.forecast_data_eval['id'].tolist())
            if self.data_loader.range_data is not None:
                for _, row in self.data_loader.range_data.iterrows():
                    documents.append(f"{row['store']} {row['category']} range: {row['range']}")
            if self.data_loader.calendar_data is not None:
                for _, row in self.data_loader.calendar_data.iterrows():
                    if pd.notna(row.get('event_name_1')):
                        event_date = row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else f"day {row['d']}"
                        documents.append(f"Date {event_date}: {row['event_name_1']} ({row['event_type_1']})")
                    if pd.notna(row.get('event_name_2')):
                        event_date = row['date'].strftime('%Y-%m-%d') if pd.notna(row.get('date')) else f"day {row['d']}"
                        documents.append(f"Date {event_date}: {row['event_name_2']} ({row['event_type_2']})")

            if documents:
                embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                self.vector_store = FAISS.from_texts(documents, embeddings)
        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")

    def _validate_date_range(self, start_date: str, end_date: str) -> Optional[Tuple[datetime, datetime]]:
        """Validate and parse date range in YYYY-MM-DD or DD-MM-YYYY format (for chart only)"""
        try:
            start_dt = pd.to_datetime(start_date, dayfirst=False, errors='coerce')
            end_dt = pd.to_datetime(end_date, dayfirst=False, errors='coerce')
            if pd.isna(start_dt) or pd.isna(end_dt):
                st.error(f"Invalid date format for {start_date} or {end_date}. Use YYYY-MM-DD.")
                return None
            if start_dt > end_dt:
                st.error(f"Start date {start_date} is after end date {end_date}.")
                return None

            # Allowed forecast window for charting
            eval_min = pd.to_datetime("2016-05-23")
            eval_max = pd.to_datetime("2016-06-19")
            if start_dt < eval_min or end_dt > eval_max:
                st.error(f"Forecast must be between {eval_min.strftime('%Y-%m-%d')} and {eval_max.strftime('%Y-%m-%d')}")
                return None

            return (start_dt, end_dt)
        except Exception as e:
            st.error(f"Error parsing dates: {str(e)}")
            return None

    def parse_query(self, query: str, prev_query: Optional[Dict] = None) -> Dict:
        """Parse user query using LLM to extract category, item, store(s), and date range"""
        prompt_template = PromptTemplate(
            input_variables=["query", "prev_query", "categories", "stores"],
            template="""
            You are a retail forecasting assistant. Parse the user's query to extract:
            - Category (one of: {categories}, or simplified as HOBBIES, HOUSEHOLD, FOODS, or specific sub-categories like HOBBIES_1, FOODS_1)
            - Item number (three-digit number, e.g., '001', default '001' if not specified)
            - Store(s) (one or more of: {stores}, or state codes CA, TX, WI for all stores in that state, or 'all stores')
            - Date range (format 'YYYY-MM-DD to YYYY-MM-DD' or 'DD-MM-YYYY to DD-MM-YYYY', or None if not specified)

            Map simplified categories to data categories only if no specific sub-category is provided:
            - HOBBIES -> HOBBIES_1 or HOBBIES_2 (default to HOBBIES_1 if item starts with '1', else HOBBIES_2)
            - HOUSEHOLD -> HOUSEHOLD_1 or HOUSEHOLD_2 (default to HOUSEHOLD_1 if item starts with '1', else HOUSEHOLD_2)
            - FOODS -> FOODS_1, FOODS_2, or FOODS_3 (default to FOODS_3 for items like '001')

            If the query specifies a sub-category (e.g., FOODS_1_001 or FOODS_1), preserve it as the category unless invalid.
            If the query references a previous query (e.g., "instead of CA_2 show TX_1" or "add TX_1 and remove CA_2"):
            - Use the previous query to fill in missing fields (category, item, date range).
            - For stores, start with the previous query's stores if available.
            - If 'add' or synonym of 'add' word is in the query, include new stores in the store list.
            - If 'remove' or synonym of 'remove' word is in the query, exclude specified stores from the store list.

            If a state code (CA, TX, WI) is provided, include all stores in that state (e.g., CA -> CA_1, CA_2, CA_3, CA_4).

            Return a JSON object with 'category', 'item', 'stores', 'date_range', 'error' (null if valid, else error message).

            Query: {query}
            Previous Query: {prev_query}
            """
        )

        all_stores = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
        data_categories = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']
        state_to_stores = {
            'CA': ['CA_1', 'CA_2', 'CA_3', 'CA_4'],
            'TX': ['TX_1', 'TX_2', 'TX_3'],
            'WI': ['WI_1', 'WI_2', 'WI_3']
        }

        chain = RunnableSequence(prompt_template | self.llm | JsonOutputParser())

        try:
            response = chain.invoke({
                "query": query,
                "prev_query": json.dumps(prev_query) if prev_query else "None",
                "categories": ", ".join(['HOBBIES', 'HOUSEHOLD', 'FOODS'] + data_categories),
                "stores": ", ".join(all_stores + ['CA', 'TX', 'WI', 'all stores'])
            })
            parsed = response

            if parsed.get('error'):
                return parsed

            # Fill missing fields from prev_query
            if prev_query and not parsed.get('error'):
                if not parsed.get('category') and prev_query.get('category'):
                    parsed['category'] = prev_query['category']
                if not parsed.get('item') and prev_query.get('item'):
                    parsed['item'] = prev_query['item']
                if not parsed.get('date_range') and prev_query.get('date_range'):
                    parsed['date_range'] = prev_query['date_range']
                if not parsed.get('stores') and prev_query.get('stores'):
                    parsed['stores'] = prev_query['stores']

            # Category normalization
            item = parsed.get('item', '001')
            data_categories_set = set(data_categories)
            if parsed.get('category'):
                category_input = parsed['category'].upper()
                if category_input in data_categories_set:
                    parsed['category'] = category_input
                elif category_input == 'HOBBIES':
                    parsed['category'] = 'HOBBIES_1' if item.startswith('1') else 'HOBBIES_2'
                elif category_input == 'HOUSEHOLD':
                    parsed['category'] = 'HOUSEHOLD_1' if item.startswith('1') else 'HOUSEHOLD_2'
                elif category_input == 'FOODS':
                    parsed['category'] = 'FOODS_3'  # default for 001-like
                else:
                    parsed['error'] = (
                        f"Invalid category: {parsed['category']}. "
                        f"Must be one of: HOBBIES, HOUSEHOLD, FOODS, or sub-categories like HOBBIES_1, FOODS_1"
                    )
                    return parsed

            if not parsed.get('category') or parsed['category'] not in data_categories_set:
                parsed['error'] = (
                    "Invalid or missing category. Must be one of: "
                    "HOBBIES, HOUSEHOLD, FOODS, or sub-categories like HOBBIES_1, FOODS_1"
                )
                return parsed

            # Stores handling and state expansion
            base_stores = prev_query['stores'] if prev_query and prev_query.get('stores') else []
            if not parsed.get('stores'):
                parsed['stores'] = base_stores if base_stores else all_stores
            elif isinstance(parsed['stores'], str):
                store_upper = parsed['stores'].upper()
                if store_upper in state_to_stores:
                    parsed['stores'] = state_to_stores[store_upper]
                elif parsed['stores'].lower() == 'all stores':
                    parsed['stores'] = all_stores
                else:
                    parsed['stores'] = [parsed['stores']]
            elif isinstance(parsed['stores'], list):
                expanded_stores = []
                for store in parsed['stores']:
                    su = str(store).upper()
                    if su in state_to_stores:
                        expanded_stores.extend(state_to_stores[su])
                    else:
                        expanded_stores.append(store)
                parsed['stores'] = list(set(expanded_stores))

            # Add/remove logic relative to prev_query
            # Add/remove logic relative to prev_query
            if prev_query and base_stores:
                ql = query.lower()
                final_stores = set(base_stores)

                # Add
                if "add" in ql:
                    for store in parsed.get("stores", []):
                        su = store.upper()
                        if su in state_to_stores:
                            final_stores.update(state_to_stores[su])
                        else:
                            final_stores.add(store)

                # Remove
                if "remove" in ql:
                    for store in all_stores:
                        if f"remove {store.lower()}" in ql or f"remove {store}" in ql:
                            final_stores.discard(store)
                    for state in state_to_stores:
                        if f"remove {state.lower()}" in ql or f"remove {state}" in ql:
                            for s in state_to_stores[state]:
                                final_stores.discard(s)

                parsed['stores'] = sorted(final_stores)


            # Validate stores
            for store in parsed['stores']:
                if store not in all_stores:
                    parsed['error'] = f"Invalid store: {store}. Must be one of: {', '.join(all_stores)}"
                    return parsed

            # Item validation
            if not parsed.get('item') or not re.match(r'^\d{3}$', str(parsed['item'])):
                parsed['item'] = '001'

            # Date range validation (chart only)
            if parsed.get('date_range') and ' to ' in parsed['date_range']:
                start_date, end_date = parsed['date_range'].split(' to ')
                date_range = self._validate_date_range(start_date, end_date)
                if not date_range:
                    parsed['error'] = "Invalid date range"
                    return parsed

            return parsed
        except Exception as e:
            return {'error': f"Error parsing query: {str(e)}"}

    def get_forecast_data(
        self,
        category: str,
        item: str,
        store: str,
        date_range: Optional[Tuple[datetime, datetime]]
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve forecast data for a specific product and store, filtered by date range (for chart).
        Calculations (EOQ/SS/ROP) ignore this range and use full rows.
        """
        try:
            product_id_val = f"{category}_{item}_{store}_validation"
            product_id_eval = f"{category}_{item}_{store}_evaluation"

            forecast_data = pd.DataFrame()

            if self.data_loader.forecast_data_val is not None:
                val_data = self.data_loader.forecast_data_val[
                    self.data_loader.forecast_data_val['id'] == product_id_val
                ]
                if not val_data.empty:
                    forecast_data = val_data

            if self.data_loader.forecast_data_eval is not None:
                eval_data = self.data_loader.forecast_data_eval[
                    self.data_loader.forecast_data_eval['id'] == product_id_eval
                ]
                if not eval_data.empty:
                    forecast_data = pd.concat([forecast_data, eval_data], axis=1)
                    if forecast_data.columns.duplicated().any():
                        forecast_data = forecast_data.loc[:, ~forecast_data.columns.duplicated()]

            if forecast_data.empty:
                return None

            # For chart display only (optional range filter)
            if self.data_loader.calendar_data is not None and date_range:
                start_dt, end_dt = date_range
                history_days = self.data_loader.calendar_data[
                    self.data_loader.calendar_data['date'] < pd.to_datetime("2016-05-22")
                ]['d'].astype(str).tolist()

                forecast_days = self.data_loader.calendar_data[
                    (self.data_loader.calendar_data['date'] >= start_dt) &
                    (self.data_loader.calendar_data['date'] <= end_dt)
                ]['d'].astype(str).tolist()

                all_days = [d for d in history_days + forecast_days if d in forecast_data.columns]
                if not all_days:
                    return None
                forecast_data = forecast_data[['id'] + all_days]

            return forecast_data
        except Exception as e:
            st.error(f"Error retrieving forecast data: {str(e)}")
            return None

    def create_forecast_chart(self, data_dict, category, item, stores, date_range):
        try:
            fig = go.Figure()
            split_date = pd.to_datetime('2016-05-22')  # boundary between history & forecast

            # Map days to dates
            date_map = {}
            if self.data_loader.calendar_data is not None:
                for _, row in self.data_loader.calendar_data.iterrows():
                    if pd.notna(row['d']) and pd.notna(row['date']):
                        date_map[str(row['d'])] = row['date']

            # --- Plot store lines ---
            for store, data in data_dict.items():
                all_day_cols = [col for col in data.columns if col.isdigit()]
                dates = [date_map[d] for d in all_day_cols if d in date_map]
                values = [float(data[d].iloc[0]) if pd.notna(data[d].iloc[0]) else 0 for d in all_day_cols if d in date_map]

                # Separate historical and forecast
                hist_dates = [d for d in dates if d <= split_date]
                hist_values = [v for d, v in zip(dates, values) if d <= split_date]

                if date_range:
                    start_dt, end_dt = date_range
                    forecast_dates = [d for d in dates if d > split_date and start_dt <= d <= end_dt]
                    forecast_values = [v for d, v in zip(dates, values) if d > split_date and start_dt <= d <= end_dt]
                else:
                    forecast_dates = [d for d in dates if d > split_date]
                    forecast_values = [v for d, v in zip(dates, values) if d > split_date]

                # Plot historical
                if hist_dates:
                    fig.add_trace(go.Scatter(
                        x=hist_dates,
                        y=hist_values,
                        mode='lines+markers',
                        name=f"{store} (Historical)",
                        line=dict(dash='dot'),
                        marker=dict(size=6)
                    ))

                # Plot forecast
                if forecast_dates:
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        mode='lines+markers',
                        name=f"{store} (Forecast)",
                        line=dict(dash='solid'),
                        marker=dict(size=6)
                    ))

            # --- Add special day and SNAP markers with offset to avoid overlap ---
            # --- Add special day (top) and SNAP markers (bottom) ---
            # --- Add special day (top) and SNAP markers (bottom) with independent overlap control ---
            if self.data_loader.calendar_data is not None:
                cal_df = self.data_loader.calendar_data.copy()

                if date_range:
                    start_dt, end_dt = date_range
                    cal_df = cal_df[(cal_df['date'] >= start_dt) & (cal_df['date'] <= end_dt)]

                from datetime import timedelta

                # Find Y max/min for placing labels
                all_y = [y for trace in fig.data for y in trace.y if trace.y is not None]
                if not all_y:
                    all_y = [0]
                y_max = max(all_y)
                y_min = min(all_y)

                # Separate counters for top and bottom
                used_top_positions = {}    # For events
                used_bottom_positions = {} # For SNAP markers

                for _, row in cal_df.iterrows():
                    date_val = row['date']
                    if pd.isna(date_val):
                        continue

                    # --- Special events (top) ---
                    if pd.notna(row.get('event_name_1')) or pd.notna(row.get('event_name_2')):
                        top_count = used_top_positions.get(date_val, 0)
                        y_offset_top = 12 + (top_count * 14)

                        event_label = row.get('event_name_1') or row.get('event_name_2')
                        fig.add_vline(x=date_val, line=dict(color="red", dash="dash"), opacity=0.6)
                        fig.add_annotation(
                            x=date_val,
                            y=y_max,
                            text=str(event_label),
                            showarrow=False,
                            yshift=y_offset_top,
                            font=dict(size=9, color="red"),
                            textangle=30
                        )
                        used_top_positions[date_val] = top_count + 1

                    # --- SNAP markers (bottom) ---
                    snap_colors = {"snap_CA": "blue", "snap_TX": "green", "snap_WI": "purple"}
                    for snap_col, color in snap_colors.items():
                        if snap_col in row and row[snap_col] == 1:
                            bottom_count = used_bottom_positions.get(date_val, 0)
                            y_offset_bottom = -12 - (bottom_count * 14)

                            fig.add_vline(x=date_val, line=dict(color=color, dash="dot"), opacity=0.5)
                            fig.add_annotation(
                                x=date_val,
                                y=y_min,
                                text=snap_col.upper(),
                                showarrow=False,
                                yshift=y_offset_bottom,
                                font=dict(size=8, color=color),
                                textangle=30
                            )
                            used_bottom_positions[date_val] = bottom_count + 1




            fig.update_layout(
                title=f"Forecast for {category}_{item} in {', '.join(stores)}",
                xaxis_title="Date",
                yaxis_title="Sales (Units)",
                template='plotly_white',
                hovermode='x unified'
            )

            return fig
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None




    def generate_insights(self, parsed_query: Dict, forecast_data_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        category = parsed_query['category']
        item = parsed_query['item']
        stores = parsed_query['stores']

        ORDERING_COST = 50.0
        HOLDING_COST = 2.0
        Z = 1.65

        try:
            vrp_result = run_vrp_from_forecast(parsed_query, forecast_data_dict)
            if vrp_result and len(vrp_result) >= 6:
                lead_times = vrp_result[5]
            else:
                lead_times = {s: 0.0 for s in stores}
        except Exception:
            lead_times = {s: 0.0 for s in stores}

        results = []
        for store in stores:
            sku_eval = f"{category}_{item}_{store}_evaluation"
            sku_val = f"{category}_{item}_{store}_validation"

            eval_df = self.data_loader.forecast_data_eval
            eval_row = eval_df[eval_df["id"] == sku_eval] if eval_df is not None else pd.DataFrame()
            eval_vals = []
            if not eval_row.empty:
                eval_vals = pd.to_numeric(eval_row.drop(columns=["id"], errors="ignore").values.flatten(), errors="coerce")
                eval_vals = eval_vals[~np.isnan(eval_vals)]
            demand_forecast = float(np.sum(eval_vals)) if len(eval_vals) else 0.0
            avg_demand_forecast = float(np.mean(eval_vals)) if len(eval_vals) else 0.0

            val_df = self.data_loader.forecast_data_val
            val_row = val_df[val_df["id"] == sku_val] if val_df is not None else pd.DataFrame()
            val_vals = []
            if not val_row.empty:
                val_vals = pd.to_numeric(val_row.drop(columns=["id"], errors="ignore").values.flatten(), errors="coerce")
                val_vals = val_vals[~np.isnan(val_vals)]
            sigma = float(np.std(val_vals, ddof=1)) if len(val_vals) > 1 else 0.0

            lead_time_hours = float(lead_times.get(store, 0.0))
            lead_time_days = lead_time_hours / 24.0 if lead_time_hours > 0 else 0.0

            eoq = math.sqrt((2.0 * max(demand_forecast, 0.0) * ORDERING_COST) / max(HOLDING_COST, 1e-9)) if demand_forecast > 0 else 0.0
            safety_stock = Z * sigma * math.sqrt(lead_time_days) if lead_time_days > 0 else 0.0
            reorder_point = (avg_demand_forecast * lead_time_days) + safety_stock

            risk_label = "Unknown"
            if self.risk_model:
                features = [[sigma, lead_time_days, avg_demand_forecast, demand_forecast]]
                try:
                    risk_label = self.risk_model.predict(features)[0]
                except Exception:
                    pass

            results.append({
                "SKU_ID": f"{category}_{item}_{store}",
                "EOQ": round(eoq, 2),
                "ReorderPoint": round(reorder_point, 2),
                "SafetyStock": round(safety_stock, 2),
                # "RiskPrediction": risk_label
            })

        return results


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Forecasting Chatbot",
        page_icon="📈",
        layout="wide"
    )

    st.title("🤖 Forecasting Chatbot")
    st.markdown(
        "Ask about product forecasting across stores! "
        "Past (up to 2016-05-22, dotted) and future (2016-05-23 to 2016-06-19, solid) with events/SNAP markers. "
        "Use state codes (CA, TX, WI) for all stores in a state."
    )

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ForecastingChatbot()
        st.session_state.chat_history = []
        st.session_state.last_query = None
        st.session_state.vrp_map = None
        st.session_state.map_key = None
        try:
            st.session_state.chatbot.data_loader.load_data_files(
                calendar_file='calendar_forecasting_corrected.csv',
                validation_file='validation_corrected.csv',
                evaluation_file='evaluation_corrected.csv',
                range_file='result_corrected.csv'
            )
        except Exception as e:
            st.error(f"Failed to load data files: {str(e)}")

    with st.sidebar:
        st.header("📝 Example Queries")
        st.markdown("""
        • "Show forecasting of FOODS_1_001 in CA from 2016-05-23 to 2016-06-19"
        • "Show forecasting of FOODS_001 in CA from 23-05-2016 to 19-06-2016"
        • "Forecast for HOBBIES_002 in all stores"
        • "Household products forecast for TX_1"
        • "Instead of CA_2, show TX_1 for FOODS_1_001"
        • "Add TX_1 and remove CA_2"
        """)

        st.header("🏪 Available Options")
        st.markdown("""
        **Categories:** HOBBIES (HOBBIES_1, HOBBIES_2), HOUSEHOLD (HOUSEHOLD_1, HOUSEHOLD_2), FOODS (FOODS_1, FOODS_2, FOODS_3)

        **Stores:** CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3 (or CA, TX, WI)

        **Items:** 001, 002, 003, ... (default: 001)

        **Date Range (chart only):** 2016-05-23 to 2016-06-19 (YYYY-MM-DD or DD-MM-YYYY)
        """)

        if st.session_state.chatbot.data_loader.range_data is not None:
            st.header("📊 Item Ranges")
            st.dataframe(st.session_state.chatbot.data_loader.range_data, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    chart_col, map_col = st.columns([2, 1])

    with col1:
        st.header("💬 Chat with the Bot")
        user_query = st.text_input(
            "Ask about forecasting:",
            placeholder="e.g., Show forecasting of FOODS_1_001 in CA from 2016-05-23 to 2016-06-19",
            key="user_input"
        )

        if st.button("Send", type="primary") and user_query:
    # Initialize persistent store list if not present
            if "selected_stores" not in st.session_state:
                st.session_state.selected_stores = []

            # Find previous parsed query for add/remove context
            prev_query = None
            for sender, content, *rest in reversed(st.session_state.chat_history):
                if sender == "user":
                    parsed_prev = st.session_state.chatbot.parse_query(content, None)
                    if not parsed_prev.get('error'):
                        prev_query = parsed_prev
                        break

            parsed_query = st.session_state.chatbot.parse_query(user_query, prev_query)

            # Update selected_stores based on action
            q_lower = user_query.lower()
            if "add" in q_lower:
                for s in parsed_query["stores"]:
                    if s not in st.session_state.selected_stores:
                        st.session_state.selected_stores.append(s)

            elif "remove" in q_lower:
                for s in parsed_query["stores"]:
                    if s in st.session_state.selected_stores:
                        st.session_state.selected_stores.remove(s)
                    else:
                        st.warning(f"Store {s} was not already in the previous query.")

            else:
                # New base query replaces the store list
                st.session_state.selected_stores = parsed_query["stores"]

            # Always use the updated store list in parsed_query
            parsed_query["stores"] = st.session_state.selected_stores

            st.session_state.chat_history.append(("user", user_query))

            if parsed_query.get('error'):
                st.session_state.chat_history.append(("bot", parsed_query['error']))
            else:
                # Date range only for chart display (calculations ignore range)
                date_range = None
                if parsed_query.get('date_range'):
                    try:
                        start, end = parsed_query['date_range'].split(' to ')
                        date_range = st.session_state.chatbot._validate_date_range(start, end)
                        if not date_range:
                            st.session_state.chat_history.append(
                                ("bot", "Invalid date range. Use 'YYYY-MM-DD to YYYY-MM-DD' or 'DD-MM-YYYY to DD-MM-YYYY'.")
                            )
                            st.rerun()
                    except Exception:
                        st.session_state.chat_history.append(
                            ("bot", "Invalid date range format. Use 'YYYY-MM-DD to YYYY-MM-DD' or 'DD-MM-YYYY to DD-MM-YYYY'.")
                        )
                        st.rerun()

                # Optional range validation for item existence
                valid_item = False
                if st.session_state.chatbot.data_loader.range_data is not None:
                    range_check = st.session_state.chatbot.data_loader.range_data[
                        (st.session_state.chatbot.data_loader.range_data['store'].isin(parsed_query['stores'])) &
                        (st.session_state.chatbot.data_loader.range_data['category'] == parsed_query['category'])
                    ]
                    if not range_check.empty and int(parsed_query['item']) <= range_check['range'].max():
                        valid_item = True
                    else:
                        st.session_state.chat_history.append(
                            ("bot", f"Item {parsed_query['item']} is not available for {parsed_query['category']} "
                                    f"in {', '.join(parsed_query['stores'])}.")
                        )

                if valid_item or st.session_state.chatbot.data_loader.range_data is None:
                    # Build data_dict for chart & VRP
                    data_dict = {}
                    for store in parsed_query['stores']:
                        forecast_data = st.session_state.chatbot.get_forecast_data(
                            parsed_query['category'], parsed_query['item'], store, date_range
                        )
                        if forecast_data is None:
                            # Create placeholder dataframe with same structure
                            if st.session_state.chatbot.data_loader.calendar_data is not None:
                                day_cols = st.session_state.chatbot.data_loader.calendar_data['d'].astype(str).tolist()
                                zero_data = {col: [0] for col in day_cols}
                                zero_data['id'] = [f"{parsed_query['category']}_{parsed_query['item']}_{store}_evaluation"]
                                forecast_data = pd.DataFrame(zero_data)
                            else:
                                forecast_data = pd.DataFrame({"id": [f"{parsed_query['category']}_{parsed_query['item']}_{store}_evaluation"]})

                        data_dict[store] = forecast_data


                    if data_dict:
                        current_query = f"{parsed_query['category']}_{parsed_query['item']}_" \
                                        f"{'_'.join(sorted(parsed_query['stores']))}_" \
                                        f"{parsed_query.get('date_range', 'all')}"

                        # Chart
                        chart = st.session_state.chatbot.create_forecast_chart(
                            data_dict, parsed_query['category'], parsed_query['item'], parsed_query['stores'], date_range
                        )

                        # Inventory metrics (JSON) — always full rows, ignoring date_range
                        metrics = st.session_state.chatbot.generate_insights(parsed_query, data_dict)

                        # Append bot JSON to history
                        st.session_state.chat_history.append(("bot_json", metrics))

                        with chart_col:
                            if chart:
                                unique_key = f"chart_{parsed_query['category']}_{parsed_query['item']}_" \
                                             f"{'_'.join(sorted(parsed_query['stores']))}_" \
                                             f"{date_range[0].strftime('%Y%m%d') if date_range else 'all'}_" \
                                             f"{uuid.uuid4()}"
                                st.plotly_chart(chart, use_container_width=True, key=unique_key)

                        with map_col:
                            # Run VRP and render map (capture lead_times too for consistency)
                            vrp_result = run_vrp_from_forecast(parsed_query, data_dict)
                            if vrp_result:
                                # After unpacking vrp_result
                                if len(vrp_result) == 6:
                                    vrp_map, route_order, total_distance, total_time, total_cost, lead_times = vrp_result
                                else:
                                    vrp_map, route_order, total_distance, total_time, total_cost = vrp_result
                                    lead_times = {}

                                # Store VRP results in session
                                st.session_state.vrp_map = vrp_map
                                st.session_state.route_order = route_order
                                st.session_state.total_distance = total_distance
                                st.session_state.total_time = total_time
                                st.session_state.total_cost = total_cost
                                st.session_state.last_query = current_query
                                st.session_state.lead_times = lead_times  # ✅ keep lead_times

                            else:
                                vrp_map = None

                            if st.session_state.get('vrp_map'):
                                st.markdown("### 🚛 Route")
                                try:
                                    st_folium(st.session_state.vrp_map, width=600, height=400,
                                              key=st.session_state.map_key, returned_objects=[])
                                except Exception as e:
                                    st.error(f"Error rendering map: {str(e)}")
                                    st.session_state.vrp_map.save("temp_map.html")
                                    with open("temp_map.html", "r") as f:
                                        map_html = f.read()
                                    st.components.v1.html(map_html, width=400, height=400)

                                st.markdown(f"**Total Distance:** {st.session_state.total_distance:.2f} km")
                                st.markdown(f"**Total Time:** {st.session_state.total_time / 3600:.2f} hours")
                                st.markdown(f"**Total Cost:** ${st.session_state.total_cost:.2f}")
                            else:
                                st.info("No route available for selected stores.")

                            if 'exec_agent' not in st.session_state:
                                st.session_state.exec_agent = ExecutiveSummaryAgent()

                            exec_summary = st.session_state.exec_agent.generate_report(
                                parsed_query,
                                data_dict,
                                metrics,
                                vrp_result,
                                st.session_state.chatbot.data_loader,
                                lead_times=st.session_state.get("lead_times", {})  # ✅ forward lead_times
                            )


                            st.markdown("## 📊 Executive Summary")
                            st.write(exec_summary)
                            
                    else:
                        st.session_state.chat_history.append((
                            "bot",
                            f"No forecast data found for {parsed_query['category']}_{parsed_query['item']} in "
                            f"{', '.join(parsed_query['stores'])}."
                        ))

    # Conversation history (show latest ~10)
    st.header("💭 Conversation History")
    history_to_show = st.session_state.chat_history[-10:]
    for i, (sender, content, *rest) in enumerate(reversed(history_to_show)):
        if sender == "user":
            st.markdown(f"**You:** {content}")
        elif sender == "bot":
            st.markdown(f"**Bot:** {content}")
        elif sender == "bot_json":
            # Pretty JSON rendering
            st.json(content)
        elif sender == "chart":
            history_key = f"history_chart_{i}_{uuid.uuid4()}"
            st.plotly_chart(content, use_container_width=True, key=history_key)
        st.markdown("---")


if __name__ == "__main__":
    main()
