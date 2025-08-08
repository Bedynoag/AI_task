import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from data_loader import DataLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os
import json
import uuid

# Load environment variables
load_dotenv()

class ForecastingChatbot:
    """RAG-based chatbot for handling forecasting queries with context awareness"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.chat_history = []
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment. Make sure it's defined in your .env file.")
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        self.vector_store = None
        self.setup_vector_store()
    
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
                    if pd.notna(row['event_name_1']):
                        event_date = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else f"day {row['d']}"
                        documents.append(f"Date {event_date}: {row['event_name_1']} ({row['event_type_1']})")
                    if pd.notna(row['event_name_2']):
                        event_date = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else f"day {row['d']}"
                        documents.append(f"Date {event_date}: {row['event_name_2']} ({row['event_type_2']})")
            
            if documents:
                embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
                self.vector_store = FAISS.from_texts(documents, embeddings)
        except Exception as e:
            st.error(f"Error setting up vector store: {str(e)}")
    
    def _validate_date_range(self, start_date: str, end_date: str) -> Optional[Tuple[datetime, datetime]]:
        """Validate and parse date range in YYYY-MM-DD or DD-MM-YYYY format"""
        try:
            # Try YYYY-MM-DD first
            start_dt = pd.to_datetime(start_date, format='%Y-%m-%d', errors='coerce')
            end_dt = pd.to_datetime(end_date, format='%Y-%m-%d', errors='coerce')
            if pd.isna(start_dt) or pd.isna(end_dt):
                # Try DD-MM-YYYY
                start_dt = pd.to_datetime(start_date, format='%d-%m-%Y', errors='coerce')
                end_dt = pd.to_datetime(end_date, format='%d-%m-%Y', errors='coerce')
            if pd.isna(start_dt) or pd.isna(end_dt):
                st.error(f"Invalid date format for {start_date} or {end_date}. Expected: YYYY-MM-DD or DD-MM-YYYY.")
                return None
            if start_dt > end_dt:
                st.error(f"Start date {start_date} is after end date {end_date}.")
                return None
            # Ensure dates are within calendar data range
            if self.data_loader.calendar_data is not None:
                min_date = self.data_loader.calendar_data['date'].min()
                max_date = self.data_loader.calendar_data['date'].max()
                if start_dt < min_date or end_dt > max_date:
                    st.error(f"Dates must be between {min_date.strftime('%Y-%m-%d')} and {max_date.strftime('%Y-%m-%d')}.")
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
            
            For long or complex queries, focus on extracting only the relevant components (category, item, stores, date range) and ignore extraneous details unrelated to forecasting. For example:
            - Query: "Show me the forecast for FOODS_1 item 001 in California stores, but exclude CA_2, include TX_1, and focus on the period from May 23, 2016, to June 19, 2016, with insights on sales trends"
              - Category: FOODS_1
              - Item: 001
              - Stores: CA_1, CA_3, CA_4, TX_1
              - Date range: 2016-05-23 to 2016-06-19
            - Query: "I want a detailed analysis of HOBBIES_002 in TX stores for the last month of data available, please include all Texas stores and skip WI stores"
              - Category: HOBBIES
              - Item: 002
              - Stores: TX_1, TX_2, TX_3
              - Date range: None (use max available date range)
            
            If the query is out of context (e.g., unrelated to retail forecasting), return an error message.
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
        
        chain = RunnableSequence(
            prompt_template | self.llm | JsonOutputParser()
        )
        
        try:
            response = chain.invoke({
                "query": query,
                "prev_query": json.dumps(prev_query) if prev_query else "None",
                "categories": ", ".join(['HOBBIES', 'HOUSEHOLD', 'FOODS', 'HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1', 'FOODS_2', 'FOODS_3']),
                "stores": ", ".join(all_stores)
            })
            
            parsed = response
            if parsed.get('error'):
                return parsed
            
            # Validate and fill in missing fields using prev_query
            if prev_query and not parsed.get('error'):
                if not parsed.get('category') and prev_query.get('category'):
                    parsed['category'] = prev_query['category']
                if not parsed.get('item') and prev_query.get('item'):
                    parsed['item'] = prev_query['item']
                if not parsed.get('date_range') and prev_query.get('date_range'):
                    parsed['date_range'] = prev_query['date_range']
                if not parsed.get('stores') and prev_query.get('stores'):
                    parsed['stores'] = prev_query['stores']
            
            # Map simplified category to data category only if no specific sub-category is provided
            item = parsed.get('item', '001')
            if parsed.get('category'):
                category_input = parsed['category'].upper()
                if category_input in data_categories:
                    parsed['category'] = category_input  # Preserve if already a valid sub-category
                elif category_input == 'HOBBIES':
                    parsed['category'] = 'HOBBIES_1' if item.startswith('1') else 'HOBBIES_2'
                elif category_input == 'HOUSEHOLD':
                    parsed['category'] = 'HOUSEHOLD_1' if item.startswith('1') else 'HOUSEHOLD_2'
                elif category_input == 'FOODS':
                    parsed['category'] = 'FOODS_3'  # Default to FOODS_3 for FOODS_001
                elif parsed['category'] not in data_categories:
                    parsed['error'] = f"Invalid category: {parsed['category']}. Must be one of: HOBBIES, HOUSEHOLD, FOODS, or sub-categories like HOBBIES_1, FOODS_1"
                    return parsed
            
            # Validate category
            if not parsed.get('category') or parsed['category'] not in data_categories:
                parsed['error'] = f"Invalid or missing category. Must be one of: HOBBIES, HOUSEHOLD, FOODS, or sub-categories like HOBBIES_1, FOODS_1"
                return parsed
            
            # Handle stores with add/remove logic
            base_stores = prev_query['stores'] if prev_query and prev_query.get('stores') else []
            if not parsed.get('stores'):
                parsed['stores'] = base_stores if base_stores else all_stores
            elif isinstance(parsed['stores'], str):
                if parsed['stores'].upper() in state_to_stores:
                    parsed['stores'] = state_to_stores[parsed['stores'].upper()]
                elif parsed['stores'].lower() == 'all stores':
                    parsed['stores'] = all_stores
                else:
                    parsed['stores'] = [parsed['stores']]
            elif isinstance(parsed['stores'], list):
                expanded_stores = []
                for store in parsed['stores']:
                    if store.upper() in state_to_stores:
                        expanded_stores.extend(state_to_stores[store.upper()])
                    else:
                        expanded_stores.append(store)
                parsed['stores'] = list(set(expanded_stores))
            
            # Apply add/remove logic
            if prev_query and base_stores:
                if 'add' in query.lower() or 'remove' in query.lower():
                    final_stores = set(base_stores)
                    # Add new stores
                    if 'add' in query.lower():
                        final_stores.update(parsed['stores'])
                    # Remove specified stores
                    if 'remove' in query.lower():
                        stores_to_remove = []
                        for store in all_stores:
                            if f"remove {store.lower()}" in query.lower() or f"remove {store}" in query.lower():
                                stores_to_remove.append(store)
                        for state in state_to_stores:
                            if f"remove {state.lower()}" in query.lower() or f"remove {state}" in query.lower():
                                stores_to_remove.extend(state_to_stores[state])
                        final_stores.difference_update(stores_to_remove)
                    parsed['stores'] = list(final_stores) if final_stores else parsed['stores']
            
            # Validate store names
            for store in parsed['stores']:
                if store not in all_stores:
                    parsed['error'] = f"Invalid store: {store}. Must be one of: {', '.join(all_stores)}"
                    return parsed
            
            # Validate item number
            if not parsed.get('item') or not re.match(r'^\d{3}$', parsed['item']):
                parsed['item'] = '001'
            
            # Validate date range
            if parsed.get('date_range') and ' to ' in parsed['date_range']:
                start_date, end_date = parsed['date_range'].split(' to ')
                date_range = self._validate_date_range(start_date, end_date)
                if not date_range:
                    parsed['error'] = "Invalid date range"
                    return parsed
            
            return parsed
        except Exception as e:
            return {'error': f"Error parsing query: {str(e)}"}
    
    def get_forecast_data(self, category: str, item: str, store: str, date_range: Optional[Tuple[datetime, datetime]]) -> Optional[pd.DataFrame]:
        """Retrieve forecast data for a specific product and store, filtered by date range"""
        try:
            product_id_val = f"{category}_{item}_{store}_validation"
            product_id_eval = f"{category}_{item}_{store}_evaluation"
            
            # Initialize empty DataFrame
            forecast_data = pd.DataFrame()
            
            # Get validation data (up to 2016-05-22, day 1941)
            if self.data_loader.forecast_data_val is not None:
                val_data = self.data_loader.forecast_data_val[
                    self.data_loader.forecast_data_val['id'] == product_id_val
                ]
                if not val_data.empty:
                    forecast_data = val_data
            
            # Get evaluation data (2016-05-23 to 2016-06-19, days 1942-1969)
            if self.data_loader.forecast_data_eval is not None:
                eval_data = self.data_loader.forecast_data_eval[
                    self.data_loader.forecast_data_eval['id'] == product_id_eval
                ]
                if not eval_data.empty:
                    forecast_data = pd.concat([forecast_data, eval_data], axis=1)
                    # Remove duplicate 'id' column if present
                    if forecast_data.columns.duplicated().any():
                        forecast_data = forecast_data.loc[:, ~forecast_data.columns.duplicated()]
            
            if forecast_data.empty:
                return None
            
            # Filter by date range if provided
            if date_range and self.data_loader.calendar_data is not None:
                start_dt, end_dt = date_range
                calendar_subset = self.data_loader.calendar_data[
                    (self.data_loader.calendar_data['date'] >= start_dt) &
                    (self.data_loader.calendar_data['date'] <= end_dt)
                ]
                day_columns = [str(d) for d in calendar_subset['d'] if str(d) in forecast_data.columns]
                if not day_columns:
                    return None
                forecast_data = forecast_data[['id'] + day_columns]
            
            return forecast_data
        except Exception as e:
            st.error(f"Error retrieving forecast data: {str(e)}")
            return None
    
    def create_forecast_chart(self, data_dict: Dict[str, pd.DataFrame], category: str, item: str, stores: List[str], date_range: Optional[Tuple[datetime, datetime]]) -> Optional[go.Figure]:
        """Create a Plotly chart for forecast data with event and SNAP markers"""
        try:
            fig = go.Figure()
            
            # Determine day columns and corresponding dates
            all_days = []
            for data in data_dict.values():
                all_days.extend([col for col in data.columns if re.match(r'^\d+$', col)])
            all_days = sorted(list(set(all_days)))
            
            if not all_days:
                return None
            
            # Map days to dates
            date_map = {}
            if self.data_loader.calendar_data is not None:
                for _, row in self.data_loader.calendar_data.iterrows():
                    if pd.notna(row['d']) and pd.notna(row['date']):
                        date_map[str(row['d'])] = row['date']
            
            # Filter days by date range
            if date_range:
                start_dt, end_dt = date_range
                all_days = [
                    d for d in all_days
                    if d in date_map and start_dt <= date_map[d] <= end_dt
                ]
            
            # Add traces for each store
            split_date = pd.to_datetime('2016-05-22')
            for store, data in data_dict.items():
                dates = []
                values = []
                for d in all_days:
                    if d in data.columns and d in date_map:
                        dates.append(date_map[d])
                        values.append(float(data[d].iloc[0]) if pd.notna(data[d].iloc[0]) else 0)
                
                # Split into validation (up to 2016-05-22) and evaluation (after)
                val_dates = [d for d, v in zip(dates, values) if d <= split_date]
                val_values = [v for d, v in zip(dates, values) if d <= split_date]
                eval_dates = [d for d, v in zip(dates, values) if d > split_date]
                eval_values = [v for d, v in zip(dates, values) if d > split_date]
                
                # Validation trace (dotted)
                if val_dates:
                    fig.add_trace(go.Scatter(
                        x=val_dates,
                        y=val_values,
                        mode='lines+markers',
                        name=f"{store} (Past)",
                        line=dict(dash='dot'),
                        marker=dict(size=6)
                    ))
                
                # Evaluation trace (solid)
                if eval_dates:
                    fig.add_trace(go.Scatter(
                        x=eval_dates,
                        y=eval_values,
                        mode='lines+markers',
                        name=f"{store} (Forecast)",
                        line=dict(dash='solid'),
                        marker=dict(size=6)
                    ))
            
            # Add event markers with stacked annotations
            if self.data_loader.calendar_data is not None:
                events = self.data_loader.calendar_data[
                    self.data_loader.calendar_data['event_name_1'].notna() |
                    self.data_loader.calendar_data['event_name_2'].notna()
                ]
                event_dates = events.groupby('date').agg({'event_name_1': lambda x: x.dropna().tolist(), 'event_name_2': lambda x: x.dropna().tolist()}).reset_index()
                for _, row in event_dates.iterrows():
                    date = row['date']
                    if date_range and (date < start_dt or date > end_dt):
                        continue
                    event_names = [name for sublist in [row['event_name_1'], row['event_name_2']] for name in sublist if pd.notna(name)]
                    if event_names:
                        for i, event_name in enumerate(event_names):
                            y_offset = 1.1 - (i * 0.1)  # Stack annotations vertically (e.g., 1.1, 1.0, 0.9)
                            fig.add_vline(
                                x=date.timestamp() * 1000,
                                line=dict(color='red' if i == 0 else 'purple', dash='dash'),
                                annotation_text=event_name,
                                annotation_position=f'top left',
                                annotation=dict(y=y_offset, yref="paper", showarrow=False)
                            )
            
            # Add SNAP markers with stacked annotations
            if self.data_loader.calendar_data is not None:
                snap_dates = self.data_loader.calendar_data[
                    (self.data_loader.calendar_data['snap_CA'] == 1) |
                    (self.data_loader.calendar_data['snap_TX'] == 1) |
                    (self.data_loader.calendar_data['snap_WI'] == 1)
                ]
                snap_grouped = snap_dates.groupby('date').agg({
                    'snap_CA': 'sum',
                    'snap_TX': 'sum',
                    'snap_WI': 'sum'
                }).reset_index()
                for _, row in snap_grouped.iterrows():
                    date = row['date']
                    if date_range and (date < start_dt or date > end_dt):
                        continue
                    snap_states = []
                    if row['snap_CA'] > 0:
                        snap_states.append("CA")
                    if row['snap_TX'] > 0:
                        snap_states.append("TX")
                    if row['snap_WI'] > 0:
                        snap_states.append("WI")
                    if snap_states:
                        for i, state in enumerate(snap_states):
                            y_offset = 0.1 + (i * 0.1)  # Stack annotations vertically from bottom (e.g., 0.1, 0.2, 0.3)
                            fig.add_vline(
                                x=date.timestamp() * 1000,
                                line=dict(color='green', dash='dot', width=1),
                                annotation_text=f"SNAP {state}",
                                annotation_position='bottom left',
                                annotation=dict(y=y_offset, yref="paper", showarrow=False)
                            )
            
            # Update layout
            fig.update_layout(
                title=f"Forecast for {category}_{item} in {', '.join(stores)}",
                xaxis_title="Date",
                yaxis_title="Sales (Units)",
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )
            
            return fig
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
    
    def generate_insights(self, data_dict: Dict[str, pd.DataFrame], category: str, item: str, stores: List[str]) -> str:
        """Generate per-store insights from forecast data"""
        try:
            insights = []
            date_map = {}
            if self.data_loader.calendar_data is not None:
                for _, row in self.data_loader.calendar_data.iterrows():
                    if pd.notna(row['d']) and pd.notna(row['date']):
                        date_map[str(row['d'])] = row['date']
            
            for store in stores:
                if store not in data_dict:
                    insights.append(f"**{store}**: No forecast data available.")
                    continue
                
                data = data_dict[store]
                date_columns = [col for col in data.columns if re.match(r'^\d+$', col)]
                if not date_columns:
                    insights.append(f"**{store}**: No forecast data available for the specified period.")
                    continue
                
                # Collect sales data for this store
                values = []
                for d in date_columns:
                    if pd.notna(data[d].iloc[0]):
                        values.append(float(data[d].iloc[0]))
                
                if not values:
                    insights.append(f"**{store}**: No valid sales data available.")
                    continue
                
                # Calculate metrics
                avg_sales = np.mean(values)
                max_sales = np.max(values)
                min_sales = np.min(values)
                total_sales = np.sum(values)
                trend_slope = np.polyfit(range(len(values)), sorted(values), 1)[0] if len(values) > 1 else 0
                trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
                
                # Event impact for this store
                event_insight = ""
                if self.data_loader.calendar_data is not None:
                    event_days = self.data_loader.calendar_data[
                        self.data_loader.calendar_data['event_name_1'].notna() |
                        self.data_loader.calendar_data['event_name_2'].notna()
                    ]
                    event_sales = []
                    for _, row in event_days.iterrows():
                        d = str(row['d'])
                        if d in date_columns and pd.notna(data[d].iloc[0]):
                            event_sales.append(float(data[d].iloc[0]))
                    if event_sales:
                        avg_event_sales = np.mean(event_sales)
                        event_insight = f"\n• **Event Impact**: Average sales on event days: {avg_event_sales:.1f} units (vs. {avg_sales:.1f} overall)."
                
                # SNAP impact for this store
                snap_insight = ""
                state = store.split('_')[0]
                snap_col = f'snap_{state}'
                if self.data_loader.calendar_data is not None and snap_col in self.data_loader.calendar_data.columns:
                    snap_dates = self.data_loader.calendar_data[
                        self.data_loader.calendar_data[snap_col] == 1
                    ]['d'].astype(str).tolist()
                    snap_sales = []
                    for d in date_columns:
                        if d in snap_dates and pd.notna(data[d].iloc[0]):
                            snap_sales.append(float(data[d].iloc[0]))
                    if snap_sales:
                        avg_snap_sales = np.mean(snap_sales)
                        snap_insight = f"\n• **SNAP Impact ({state})**: Average sales on SNAP days: {avg_snap_sales:.1f} units (vs. {avg_sales:.1f} overall)."
                
                # Format insights for this store
                store_insights = f"""
                **{store} Insights for {category}_{item}:**
                • **Average Daily Sales**: {avg_sales:.1f} units
                • **Peak Sales**: {max_sales:.0f} units
                • **Minimum Sales**: {min_sales:.0f} units
                • **Total Forecast**: {total_sales:.0f} units over {len(date_columns)} days
                • **Trend**: Sales are {trend_direction} (slope: {trend_slope:.3f})
                {event_insight}
                {snap_insight}
                • **Sales Volatility**: {np.std(values):.2f}
                • **Non-zero Sales Days**: {len([v for v in values if v > 0])}/{len(date_columns)} days
                """
                insights.append(store_insights)
            
            if not insights:
                return "No forecast data available for insights."
            
            return "\n\n".join(insights)
        except Exception as e:
            return f"Error generating insights: {str(e)}"

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Forecasting Chatbot",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🤖 Forecasting Chatbot")
    st.markdown("Ask about product forecasting across stores! See past (up to 2016-05-22, dotted line) and future (2016-05-23 to 2016-06-19, solid line) predictions with special event and SNAP markers. Use state codes (CA, TX, WI) for all stores in a state.")
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ForecastingChatbot()
        st.session_state.chat_history = []
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
        Try asking questions like:
        • "Show forecasting of FOODS_1_001 in CA from 2016-05-23 to 2016-06-19"
        • "Show forecasting of FOODS_001 in CA from 23-05-2016 to 19-06-2016"
        • "Forecast for HOBBIES_002 in all stores"
        • "Household products forecast for TX_1"
        • "Instead of CA_2, show TX_1 for FOODS_1_001"
        • "Add TX_1 and remove CA_2"
        • "Show me the forecast for FOODS_1 item 001 in California stores, but exclude CA_2, include TX_1, and focus on the period from May 23, 2016, to June 19, 2016, with insights on sales trends"
        """)
        
        st.header("🏪 Available Options")
        st.markdown("""
        **Categories:**
        - HOBBIES (HOBBIES_1, HOBBIES_2)
        - HOUSEHOLD (HOUSEHOLD_1, HOUSEHOLD_2)
        - FOODS (FOODS_1, FOODS_2, FOODS_3)
        
        **Stores:**
        - CA_1, CA_2, CA_3, CA_4
        - TX_1, TX_2, TX_3
        - WI_1, WI_2, WI_3
        - Or use state codes: CA, TX, WI
        
        **Items:**
        - 001, 002, 003, etc. (default: 001)
        
        **Date Ranges:**
        - Past: up to 2016-05-22 (dotted line)
        - Future: 2016-05-23 to 2016-06-19 (solid line)
        - Format: YYYY-MM-DD or DD-MM-YYYY
        """)
        
        if st.session_state.chatbot.data_loader.range_data is not None:
            st.header("📊 Item Ranges")
            st.dataframe(st.session_state.chatbot.data_loader.range_data, use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with the Bot")
        
        user_query = st.text_input(
            "Ask about forecasting:",
            placeholder="e.g., Show forecasting of FOODS_1_001 in CA from 2016-05-23 to 2016-06-19",
            key="user_input"
        )
        
        if st.button("Send", type="primary") and user_query:
            prev_query = None
            for sender, content, *rest in reversed(st.session_state.chat_history):
                if sender == "user":
                    parsed_prev = st.session_state.chatbot.parse_query(content, None)
                    if not parsed_prev.get('error'):
                        prev_query = parsed_prev
                        break
            parsed_query = st.session_state.chatbot.parse_query(user_query, prev_query)
            st.session_state.chat_history.append(("user", user_query))
            
            if parsed_query.get('error'):
                st.session_state.chat_history.append(("bot", parsed_query['error']))
            else:
                date_range = None
                if parsed_query.get('date_range'):
                    try:
                        start, end = parsed_query['date_range'].split(' to ')
                        date_range = st.session_state.chatbot._validate_date_range(start, end)
                        if not date_range:
                            st.session_state.chat_history.append(("bot", "Invalid date range. Use 'YYYY-MM-DD to YYYY-MM-DD' or 'DD-MM-YYYY to DD-MM-YYYY'."))
                            st.rerun()
                    except:
                        st.session_state.chat_history.append(("bot", "Invalid date range format. Use 'YYYY-MM-DD to YYYY-MM-DD' or 'DD-MM-YYYY to DD-MM-YYYY'."))
                        st.rerun()
                
                valid_item = False
                if st.session_state.chatbot.data_loader.range_data is not None:
                    range_check = st.session_state.chatbot.data_loader.range_data[
                        (st.session_state.chatbot.data_loader.range_data['store'].isin(parsed_query['stores'])) &
                        (st.session_state.chatbot.data_loader.range_data['category'] == parsed_query['category'])
                    ]
                    if not range_check.empty and int(parsed_query['item']) <= range_check['range'].max():
                        valid_item = True
                    else:
                        st.session_state.chat_history.append(("bot", f"Item {parsed_query['item']} is not available for {parsed_query['category']} in {', '.join(parsed_query['stores'])}."))
                
                if valid_item or st.session_state.chatbot.data_loader.range_data is None:
                    data_dict = {}
                    for store in parsed_query['stores']:
                        forecast_data = st.session_state.chatbot.get_forecast_data(
                            parsed_query['category'], parsed_query['item'], store, date_range
                        )
                        if forecast_data is not None:
                            data_dict[store] = forecast_data
                    
                    if data_dict:
                        chart = st.session_state.chatbot.create_forecast_chart(
                            data_dict, parsed_query['category'], parsed_query['item'], parsed_query['stores'], date_range
                        )
                        insights = st.session_state.chatbot.generate_insights(
                            data_dict, parsed_query['category'], parsed_query['item'], parsed_query['stores']
                        )
                        st.session_state.chat_history.append(("bot", insights))
                        if chart:
                            # Generate a unique key using UUID
                            unique_key = f"chart_{parsed_query['category']}_{parsed_query['item']}_{'_'.join(sorted(parsed_query['stores']))}_{date_range[0].strftime('%Y%m%d') if date_range else 'all'}_{uuid.uuid4()}"
                            st.session_state.chat_history.append(("chart", chart, unique_key))
                    else:
                        st.session_state.chat_history.append((
                            "bot",
                            f"No forecast data found for {parsed_query['category']}_{parsed_query['item']} in {', '.join(parsed_query['stores'])}."
                        ))
        
        st.header("💭 Conversation History")
        for i, (sender, content, *key) in enumerate(reversed(st.session_state.chat_history[-10:])):
            if sender == "user":
                st.markdown(f"**You:** {content}")
            elif sender == "bot":
                st.markdown(f"**Bot:** {content}")
            elif sender == "chart":
                st.plotly_chart(content, use_container_width=True, key=key[0] if key else f"chart_{i}")
            st.markdown("---")
    
    with col2:
        st.header("📊 Quick Stats")
        if st.session_state.chatbot.data_loader.forecast_data_val is not None:
            total_products = len(st.session_state.chatbot.data_loader.forecast_data_val)
            categories = set(st.session_state.chatbot.data_loader.forecast_data_val['id'].str.split('_').str[0])
            st.metric("Total Products", total_products)
            st.metric("Categories", len(categories))
            st.metric("Stores", 10)
            
            st.header("📋 Sample Data")
            sample_data = st.session_state.chatbot.data_loader.forecast_data_val[['id']].head()
            st.dataframe(sample_data, use_container_width=True)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()