import pyodbc
import streamlit as st

def get_connection():
    try:
        conn = pyodbc.connect(
            'Driver={ODBC Driver 18 for SQL Server};'
            'Server=tcp:rushi.database.windows.net,1433;'
            'Database=sourcehomeloan;'
            'Uid=RushiServer;'
            'Pwd=Rutuja492001;'
            'Encrypt=yes;'
            'TrustServerCertificate=yes;'
            'Connection Timeout=30;'
        )
        return conn
    except pyodbc.Error as err:
        st.error(f"❌ SQL Server Connection Error: {err}")
        return None
