import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import io

class DataLoader:
    """Class to handle loading and processing of calendar, forecast, and range data"""
    
    def __init__(self):
        self.calendar_data = None
        self.forecast_data_val = None
        self.forecast_data_eval = None
        self.range_data = None
    
    def load_calendar_data(self, calendar_file: str | io.BytesIO) -> bool:
        """Load calendar data from CSV and process date information"""
        try:
            if isinstance(calendar_file, str):
                self.calendar_data = pd.read_csv(calendar_file)
            else:
                self.calendar_data = pd.read_csv(io.StringIO(calendar_file.getvalue().decode('utf-8')))
            
            self.calendar_data.columns = self.calendar_data.columns.str.strip()
            
            # Convert 'date' column to datetime
            if 'date' in self.calendar_data.columns:
                try:
                    self.calendar_data['date'] = pd.to_datetime(self.calendar_data['date'], format='%Y-%m-%d', errors='coerce')
                    if self.calendar_data['date'].isna().any():
                        st.warning("Some dates in calendar data could not be parsed. Continuing with valid dates.")
                except Exception as e:
                    st.warning(f"Failed to convert dates in calendar data: {str(e)}. Continuing without date conversion.")
            else:
                st.warning("No 'date' column found in calendar data. Continuing without date conversion.")
            
            # Validate required columns
            required_columns = ['date', 'd', 'event_name_1', 'event_type_1', 'snap_CA', 'snap_TX', 'snap_WI']
            missing_columns = [col for col in required_columns if col not in self.calendar_data.columns]
            if missing_columns:
                st.warning(f"Calendar data missing columns: {', '.join(missing_columns)}. Some features may be limited.")
            
            return True
        except Exception as e:
            st.error(f"Error loading calendar data: {str(e)}")
            self.calendar_data = None
            return False
    
    def load_forecast_data(self, validation_file: str | io.BytesIO, evaluation_file: str | io.BytesIO) -> bool:
        """Load validation and evaluation forecast data from CSV"""
        try:
            # Load validation data (CSV)
            if isinstance(validation_file, str):
                self.forecast_data_val = pd.read_csv(validation_file)
            else:
                self.forecast_data_val = pd.read_csv(io.StringIO(validation_file.getvalue().decode('utf-8')))
                
            # Load evaluation data (CSV)
            if isinstance(evaluation_file, str):
                self.forecast_data_eval = pd.read_csv(evaluation_file)
            else:
                self.forecast_data_eval = pd.read_csv(io.StringIO(evaluation_file.getvalue().decode('utf-8')))
            
            self.forecast_data_val.columns = self.forecast_data_val.columns.astype(str).str.strip()
            self.forecast_data_eval.columns = self.forecast_data_eval.columns.astype(str).str.strip()
            
            # Validate required columns
            for df, name in [(self.forecast_data_val, 'validation'), (self.forecast_data_eval, 'evaluation')]:
                if 'id' not in df.columns:
                    st.error(f"'id' column missing in {name} data.")
                    setattr(self, f'forecast_data_{name[:3]}', None)
                    return False
                # Ensure day columns are numeric
                day_columns = [col for col in df.columns if re.match(r'^\d+$', col)]
                if not day_columns:
                    st.error(f"No day columns found in {name} data.")
                    setattr(self, f'forecast_data_{name[:3]}', None)
                    return False
            
            return True
        except Exception as e:
            st.error(f"Error loading forecast data: {str(e)}")
            self.forecast_data_val = None
            self.forecast_data_eval = None
            return False
    
    def load_range_data(self, range_file: str | io.BytesIO) -> bool:
        """Load range data from CSV"""
        try:
            if isinstance(range_file, str):
                self.range_data = pd.read_csv(range_file)
            else:
                self.range_data = pd.read_csv(io.StringIO(range_file.getvalue().decode('utf-8')))
            
            self.range_data.columns = self.range_data.columns.str.strip()
            
            # Validate required columns
            required_columns = ['store', 'category', 'range']
            missing_columns = [col for col in required_columns if col not in self.range_data.columns]
            if missing_columns:
                st.error(f"Range data missing columns: {', '.join(missing_columns)}.")
                self.range_data = None
                return False
            
            # Convert range to numeric
            self.range_data['range'] = pd.to_numeric(self.range_data['range'], errors='coerce')
            if self.range_data['range'].isna().any():
                st.warning("Some range values could not be converted to numeric. Invalid rows will be ignored.")
            
            return True
        except Exception as e:
            st.error(f"Error loading range data: {str(e)}")
            self.range_data = None
            return False
    
    def load_data_files(self, calendar_file: str | io.BytesIO, validation_file: str | io.BytesIO, 
                       evaluation_file: str | io.BytesIO, range_file: str | io.BytesIO) -> bool:
        """Load all data files"""
        calendar_success = self.load_calendar_data(calendar_file)
        forecast_success = self.load_forecast_data(validation_file, evaluation_file)
        range_success = self.load_range_data(range_file)
        
        if not (calendar_success and forecast_success and range_success):
            st.error("One or more data files failed to load. Please check the files and try again.")
            return False
        return True
    
    def get_available_products(self) -> List[str]:
        """Get list of available products from forecast data"""
        products = []
        if self.forecast_data_val is not None and 'id' in self.forecast_data_val.columns:
            products.extend(self.forecast_data_val['id'].tolist())
        if self.forecast_data_eval is not None and 'id' in self.forecast_data_eval.columns:
            products.extend(self.forecast_data_eval['id'].tolist())
        return list(set(products))
    
    def parse_product_id(self, product_id: str) -> Dict:
        """Parse product ID to extract components"""
        parts = product_id.split('_')
        if len(parts) >= 4:
            return {
                'category': parts[0],
                'dept': parts[1],
                'item': parts[2],
                'store': parts[3],
                'type': parts[4] if len(parts) > 4 else 'validation'
            }
        return {
            'category': None,
            'dept': None,
            'item': None,
            'store': None,
            'type': None
        }