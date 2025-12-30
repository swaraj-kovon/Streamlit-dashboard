# ========================
# app.py — FULL FINAL VERSION
# ========================
import os
import re
import sqlite3
import bcrypt
from datetime import datetime, date, time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

# ========================
# CONFIG
# ========================
USER_ID_COLUMN = "userId"   # change if needed

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
    with sqlite3.connect("auth.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def user_exists(email):
    with sqlite3.connect("auth.db") as conn:
        return conn.execute(
            "SELECT 1 FROM users WHERE email = ?", (email,)
        ).fetchone() is not None

def create_user(email, password):
    pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with sqlite3.connect("auth.db") as conn:
        conn.execute(
            "INSERT INTO users VALUES (?, ?, CURRENT_TIMESTAMP)",
            (email, pw)
        )

def authenticate_user(email, password):
    with sqlite3.connect("auth.db") as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
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
            st.error("Only @kovon.io emails allowed")
            return

        if len(password) < 8:
            st.error("Password must be at least 8 characters")
            return

        if mode == "Sign Up":
            if user_exists(email):
                st.error("User already exists")
                return
            create_user(email, password)
            st.success("Signup successful. Please login.")
        else:
            if not authenticate_user(email, password):
                st.error("Invalid credentials")
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
    where schemaname='public'
    order by tablename
    """
    res = supabase.rpc("run_sql", {"query": sql}).execute()
    return [r["tablename"] for r in res.data or []]

@st.cache_data(ttl=300)
def get_table_columns(table):
    sql = f"""
    select column_name
    from information_schema.columns
    where table_schema='public'
      and table_name='{table}'
    order by ordinal_position
    """
    res = supabase.rpc("run_sql", {"query": sql}).execute()
    return [r["column_name"] for r in res.data or []]

def qcol(table, col):
    return f'"{table}"."{col}"'

# =====================================================
# PAGE 1 — DATA EXPLORER
# =====================================================
def data_explorer_ui():
    st.title("📊 Data Explorer")

    if "table_results" not in st.session_state:
        st.session_state["table_results"] = {}

    tables = get_all_tables()
    selected_tables = st.multiselect("Select Tables", tables)

    if not selected_tables:
        return

    mode = st.radio(
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

            if st.button("Run Query", key=f"run_{table}"):
                df = pd.DataFrame(
                    supabase.table(table)
                    .select("*")
                    .gte(mode, start_dt.isoformat())
                    .lte(mode, end_dt.isoformat())
                    .execute()
                    .data
                )

                if mode == "updatedAt":
                    df = df[df["updatedAt"] != df["createdAt"]]

                st.session_state["table_results"][table] = df

            if table in st.session_state["table_results"]:
                df = st.session_state["table_results"][table]
                st.metric("Records", len(df))
                if not df.empty:
                    st.dataframe(df, width="stretch")
                else:
                    st.warning("No records found")

# =====================================================
# PAGE 2 — COLUMN UPDATE ANALYSIS
# =====================================================
def column_update_analysis_ui():
    st.title("🧮 Column Update Analysis")

    tables = get_all_tables()
    selected_tables = st.multiselect("Select Tables", tables)

    if not selected_tables:
        return

    mode = st.radio(
        "Mode",
        ["createdAt", "updatedAt"],
        horizontal=True
    )

    for table in selected_tables:
        with st.expander(f"📁 {table}", expanded=True):
            columns = [
                c for c in get_table_columns(table)
                if c not in ("createdAt", "updatedAt")
            ]

            if not columns:
                st.warning("No analyzable columns")
                continue

            column = st.selectbox(
                "Select Column",
                columns,
                key=f"{table}_col"
            )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                sd = st.date_input("Start Date", key=f"{table}_asd")
            with c2:
                stt = st.time_input("Start Time", key=f"{table}_ast")
            with c3:
                ed = st.date_input("End Date", key=f"{table}_aed")
            with c4:
                ett = st.time_input("End Time", key=f"{table}_aet")

            start_dt = datetime.combine(sd, stt)
            end_dt = datetime.combine(ed, ett)

            if st.button("Run Analysis", key=f"analyze_{table}"):
                df = pd.DataFrame(
                    supabase.table(table)
                    .select("*")
                    .gte(mode, start_dt.isoformat())
                    .lte(mode, end_dt.isoformat())
                    .execute()
                    .data
                )

                if mode == "updatedAt":
                    df = df[df["updatedAt"] != df["createdAt"]]

                if USER_ID_COLUMN not in df.columns:
                    st.error(f"{USER_ID_COLUMN} not found")
                    continue

                df = df[df[column].notna() & df[USER_ID_COLUMN].notna()]

                st.metric("Users affected", df[USER_ID_COLUMN].nunique())

                if not df.empty:
                    st.dataframe(
                        df[[USER_ID_COLUMN, "createdAt", "updatedAt", column]],
                        width="stretch"
                    )
                else:
                    st.warning("No matching records")

# =====================================================
# PAGE 3 — JOINS
# =====================================================
def joins_ui():
    st.title("🔗 SQL Joins Builder")

    tables = get_all_tables()
    base_table = st.selectbox("Base Table", tables)

    join_count = st.number_input(
        "Number of Joins",
        min_value=1,
        max_value=5,
        value=1
    )

    joins = []

    # -------------------------
    # JOIN DEFINITIONS
    # -------------------------
    for i in range(join_count):
        st.subheader(f"Join #{i + 1}")
        c1, c2, c3, c4 = st.columns(4)

        join_type = c1.selectbox(
            "Join Type",
            ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN"],
            key=f"jt_{i}"
        )

        join_table = c2.selectbox(
            "Join Table",
            [t for t in tables if t != base_table],
            key=f"jtab_{i}"
        )

        base_col = c3.selectbox(
            "Base Column",
            get_table_columns(base_table),
            key=f"bcol_{i}"
        )

        join_col = c4.selectbox(
            "Join Column",
            get_table_columns(join_table),
            key=f"jcol_{i}"
        )

        joins.append((join_type, join_table, base_col, join_col))

    st.divider()

    # -------------------------
    # COLUMN SELECTION
    # -------------------------
    select_columns = []
    all_tables = [base_table] + [j[1] for j in joins]

    for t in all_tables:
        chosen = st.multiselect(
            f"Select columns from {t}",
            get_table_columns(t),
            key=f"sel_{t}"
        )
        for c in chosen:
            select_columns.append((t, c))

    if not select_columns:
        st.warning("Select at least one column")
        return

    # -------------------------
    # RUN QUERY (ONCE)
    # -------------------------
    if st.button("Run Join Query"):
        select_sql = ", ".join(qcol(t, c) for t, c in select_columns)
        sql = f'SELECT {select_sql} FROM "{base_table}"'

        for jt, jt_table, bc, jc in joins:
            sql += (
                f' {jt} "{jt_table}" '
                f'ON "{base_table}"."{bc}" = "{jt_table}"."{jc}"'
            )

        with st.spinner("Executing join query..."):
            res = supabase.rpc("run_sql", {"query": sql}).execute()
            st.session_state["join_df"] = pd.DataFrame(res.data or [])

    # -------------------------
    # DISPLAY + UNIQUE FILTER
    # -------------------------
    if "join_df" not in st.session_state:
        return

    df = st.session_state["join_df"]

    st.divider()

    unique_only = st.radio(
        "Show unique only",
        ["No", "Yes"],
        horizontal=True,
        key="unique_only"
    )

    filtered_df = df

    if unique_only == "Yes" and not df.empty:
        unique_column = st.selectbox(
            "Select column",
            df.columns.tolist(),
            key="unique_column"
        )

        filtered_df = df.drop_duplicates(subset=[unique_column])

    # -------------------------
    # RESULTS
    # -------------------------
    st.metric("Rows returned", len(filtered_df))
    st.dataframe(filtered_df, width="stretch")

# =====================================================
# MAIN
# =====================================================
def main():
    init_auth_db()

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        auth_ui()
        return

    st.sidebar.success(f"Logged in as {st.session_state['email']}")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Data Explorer", "🧮 Column Update Analysis", "🔗 Joins"]
    )

    if page == "📊 Data Explorer":
        data_explorer_ui()
    elif page == "🧮 Column Update Analysis":
        column_update_analysis_ui()
    else:
        joins_ui()

if __name__ == "__main__":
    main()
