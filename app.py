import os
import re
import sqlite3
import bcrypt
from datetime import datetime, date, time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# ------------------------
# STREAMLIT CONFIG
# ------------------------
st.set_page_config(
    page_title="Kovon Data Explorer",
    layout="wide",
)

# ------------------------
# ENV SETUP
# ------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY
)

# ------------------------
# SQLITE AUTH
# ------------------------
def init_auth_db():
    conn = sqlite3.connect("auth.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def user_exists(email: str) -> bool:
    conn = sqlite3.connect("auth.db")
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE email = ?", (email,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists

def create_user(email: str, password: str):
    password_hash = bcrypt.hashpw(
        password.encode(),
        bcrypt.gensalt()
    ).decode()

    conn = sqlite3.connect("auth.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (email, password_hash)
    )
    conn.commit()
    conn.close()

def authenticate_user(email: str, password: str) -> bool:
    conn = sqlite3.connect("auth.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT password_hash FROM users WHERE email = ?",
        (email,)
    )
    row = cur.fetchone()
    conn.close()

    return bool(row and bcrypt.checkpw(password.encode(), row[0].encode()))

# ------------------------
# AUTH UI
# ------------------------
def auth_ui():
    st.title("🔐 Kovon Internal Dashboard")

    mode = st.radio("Mode", ["Sign Up", "Login"], horizontal=True)
    email = st.text_input("Company Email")
    password = st.text_input("Password", type="password")

    if st.button(mode):
        if not re.match(r"^[^@]+@kovon\.io$", email):
            st.error("Only @kovon.io emails are allowed")
            return

        if len(password) < 8:
            st.error("Password must be at least 8 characters")
            return

        if mode == "Sign Up":
            if user_exists(email):
                st.error("User already exists. Please login.")
                return
            create_user(email, password)
            st.success("Signup successful. Please login.")

        else:
            if not authenticate_user(email, password):
                st.error("Invalid email or password")
                return

            st.session_state["authenticated"] = True
            st.session_state["email"] = email
            st.rerun()

# ------------------------
# SUPABASE HELPERS
# ------------------------
@st.cache_data(ttl=300)
def get_all_tables():
    sql = """
    select tablename
    from pg_tables
    where schemaname = 'public'
    order by tablename
    """

    result = supabase.rpc("run_sql", {"query": sql}).execute()
    return [row["tablename"] for row in result.data or []]

def fetch_table_data(table, time_column, start_dt, end_dt):
    response = (
        supabase
        .table(table)
        .select("*", count="exact")
        .gte(time_column, start_dt.isoformat())
        .lte(time_column, end_dt.isoformat())
        .execute()
    )

    return pd.DataFrame(response.data), response.count or 0

# ------------------------
# DASHBOARD UI
# ------------------------
def dashboard_ui():
    st.sidebar.success(f"Logged in as {st.session_state['email']}")

    st.title("📊 Data Explorer")

    if "table_results" not in st.session_state:
        st.session_state["table_results"] = {}

    tables = get_all_tables()

    selected_tables = st.multiselect("Select Tables", tables)
    if not selected_tables:
        return

    time_column = st.radio(
        "Filter column",
        ["createdAt", "updatedAt"],
        horizontal=True
    )

    for table in selected_tables:
        with st.expander(f"📁 {table}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                sd = st.date_input("Start Date", key=f"{table}_sd")
            with c2:
                stt = st.time_input("Start Time", key=f"{table}_st")
            with c3:
                ed = st.date_input("End Date", key=f"{table}_ed")
            with c4:
                ett = st.time_input("End Time", key=f"{table}_et")

            start_dt = datetime.combine(sd, stt)
            end_dt = datetime.combine(ed, ett)

            if st.button(f"Run Query — {table}", key=f"run_{table}"):
                df, count = fetch_table_data(table, time_column, start_dt, end_dt)
                st.session_state["table_results"][table] = {
                    "df": df,
                    "count": count
                }

            if table in st.session_state["table_results"]:
                res = st.session_state["table_results"][table]
                st.metric(f"{table} — Record Count", res["count"])

                if res["df"].empty:
                    st.warning("No records found")
                else:
                    st.dataframe(res["df"], width="stretch")

# ------------------------
# MAIN
# ------------------------
def main():
    init_auth_db()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        auth_ui()
    else:
        dashboard_ui()

if __name__ == "__main__":
    main()
