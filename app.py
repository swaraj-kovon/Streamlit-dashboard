import streamlit as st
from supabase import create_client, Client
import pandas as pd
from dotenv import load_dotenv
import os
import sqlite3
import bcrypt
from typing import List, Dict, Any, Sequence, cast

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")
if SUPABASE_URL is None:
    raise ValueError("SUPABASE_URL is missing")
if SUPABASE_KEY is None:
    raise ValueError("SUPABASE_SERVICE_ROLE or SUPABASE_KEY is missing")
SUPABASE_URL = cast(str, SUPABASE_URL)
SUPABASE_KEY = cast(str, SUPABASE_KEY)

DB_PATH = "users.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def create_user(email: str, password: str) -> bool:
    if not email.endswith("@kovon.io"):
        return False
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        cur.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    stored_hash = row[0]
    return bcrypt.checkpw(password.encode(), stored_hash)


init_db()


@st.cache_resource
def get_supabase() -> Client:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE") or os.getenv("SUPABASE_KEY")
    if supabase_url is None or supabase_key is None:
        raise ValueError("Supabase env variables missing")
    supabase_url = cast(str, supabase_url)
    supabase_key = cast(str, supabase_key)
    return create_client(supabase_url, supabase_key)


supabase: Client = get_supabase()


def list_tables() -> List[str]:
    try:
        result = supabase.rpc("get_public_tables").execute()
        raw_data = result.data
        if not isinstance(raw_data, list):
            return []
        cleaned_rows = [row for row in raw_data if isinstance(row, dict)]
        return [row["name"] for row in cleaned_rows if "name" in row]  # type: ignore
    except Exception as e:
        st.error(f"Error fetching table names: {e}")
        return []


def fetch_table(table_name: str) -> List[Dict[str, Any]]:
    try:
        result = supabase.table(table_name).select("*").execute()
        raw = result.data
        if not isinstance(raw, Sequence):
            return []
        return [row for row in raw if isinstance(row, dict)]
    except Exception as e:
        st.error(f"Error fetching data from {table_name}: {e}")
        return []


def run_custom_sql(sql: str) -> List[Dict[str, Any]]:
    try:
        result = supabase.rpc("run_sql", {"query": sql}).execute()
        raw = result.data or []
        if not isinstance(raw, Sequence):
            return []
        return [row for row in raw if isinstance(row, dict)]
    except Exception as e:
        st.error(f"Error running SQL: {e}")
        return []


def login_page():
    st.title("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(email, password):
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.rerun()
        else:
            st.error("Invalid credentials")


def signup_page():
    st.title("📝 Create Account")
    email = st.text_input("Email (must be @kovon.io)")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Sign Up"):
        if not email.endswith("@kovon.io"):
            st.error("Only @kovon.io emails allowed")
            return
        if password != confirm:
            st.error("Passwords do not match")
            return
        if create_user(email, password):
            st.success("Account created. You can now login.")
        else:
            st.error("User exists or invalid email")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "mode" not in st.session_state:
    st.session_state.mode = "dashboard"


if not st.session_state.logged_in:
    choice = st.sidebar.radio("Authentication", ["Login", "Sign Up"])
    if choice == "Login":
        login_page()
    else:
        signup_page()
    st.stop()

page = st.sidebar.radio("Navigation", ["Dashboard", "Analytics"])


#                           ANALYTICS

#                           ANALYTICS

  #                           ANALYTICS

if page == "Analytics":
    st.title("📈 Analytics")

    tables = list_tables()
    if not tables:
        st.error("No tables available.")
        st.stop()

    @st.cache_data
    def get_table_df(table_name: str) -> pd.DataFrame:
        rows = fetch_table(table_name)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)

        # Auto-detect datetime fields and convert to UTC-aware datetime64
        for col in df.columns:
            if any(k in col.lower() for k in ["updated", "created", "timestamp", "date"]):
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        return df

    def pick_column(candidates: list[str], cols: list[str]) -> str | None:
        col_set = set(cols)
        for c in candidates:
            if c in col_set:
                return c
        lower_map = {c.lower(): c for c in cols}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    # ---------- RULE STATE SETUP ----------
    if "analytics_rules" not in st.session_state:
        st.session_state.analytics_rules = [
            {
                "table": tables[0],
                "column": None,
                "operator": "All",   # default
                "value": None,
                "operation_filter": "All",
                "date_col": None,
                "start_dt": None,
                "end_dt": None,
            }
        ]

    rules = st.session_state.analytics_rules

    def add_rule():
        rules.append(
            {
                "table": tables[0],
                "column": None,
                "operator": "All",
                "value": None,
                "operation_filter": "All",
                "date_col": None,
                "start_dt": None,
                "end_dt": None,
            }
        )

    st.markdown("### Rules")

    add_col, _ = st.columns([0.2, 0.8])
    if add_col.button("➕ Add Rule"):
        add_rule()

    new_rules: list[dict[str, Any]] = []

    # ---------- RENDER EACH RULE ----------
    for idx, rule in enumerate(rules):
        st.markdown(f"#### Rule {idx + 1}")
        c_table, c_remove = st.columns([0.9, 0.1])

        # Table selector
        selected_table = c_table.selectbox(
            "Table",
            options=tables,
            index=tables.index(rule.get("table", tables[0])),
            key=f"rule_table_{idx}",
        )

        remove_rule = c_remove.button("❌", key=f"rule_remove_{idx}")

        df_tbl = get_table_df(selected_table)
        if df_tbl.empty:
            st.warning(f"No data in `{selected_table}`.")
            if not remove_rule:
                new_rules.append(rule | {"table": selected_table})
            continue

        cols = df_tbl.columns.tolist()

        # ---- Column / Operator / Value ----
        c1, c2, c3 = st.columns([2, 2, 3])

        selected_column = c1.selectbox(
            "Column",
            options=cols,
            index=cols.index(rule.get("column")) if rule.get("column") in cols else 0,
            key=f"rule_col_{idx}",
        )

        operators = ["All", "equals", "not equals", "contains"]
        selected_operator = c2.selectbox(
            "Operator",
            operators,
            index=operators.index(rule.get("operator", "All")),
            key=f"rule_op_{idx}",
        )

        value_options = (
            df_tbl[selected_column].dropna().astype(str).sort_values().unique().tolist()
        )

        if value_options and selected_operator != "All":
            default_val = rule.get("value", value_options[0])
            selected_value = c3.selectbox(
                "Value",
                options=value_options,
                index=value_options.index(default_val)
                if default_val in value_options
                else 0,
                key=f"rule_val_{idx}",
            )
        else:
            c3.write("All values")
            selected_value = None

        # ---- Operation filter ----
        op_col = (
            "operation"
            if "operation" in cols
            else pick_column(["operation"], cols)
        )

        c4, c5 = st.columns([1, 2])
        if op_col:
            op_vals = (
                df_tbl[op_col].dropna().astype(str).sort_values().unique().tolist()
            )
            selected_op_filter = c4.selectbox(
                "Operation",
                ["All"] + op_vals,
                index=(
                    ["All"] + op_vals
                ).index(rule.get("operation_filter", "All")),
                key=f"rule_operation_filter_{idx}",
            )
        else:
            c4.write("Operation\n(not available)")
            selected_op_filter = "All"

        # ---- Date range detection ----
        date_col_lower = pick_column(
            ["updatedat", "updated_at", "createdat", "created_at", "timestamp"],
            [c.lower() for c in cols],
        )
        real_date_col = (
            [c for c in cols if c.lower() == date_col_lower][0]
            if date_col_lower
            else None
        )

        # ---- Date inputs ----
        if real_date_col:
            df_tbl[real_date_col] = pd.to_datetime(df_tbl[real_date_col], errors="coerce", utc=True)
            if df_tbl[real_date_col].notna().any():
                min_ts = df_tbl[real_date_col].min()
                max_ts = df_tbl[real_date_col].max()

                d1, d2 = c5.columns(2)

                start_date = d1.date_input(
                    "Start date",
                    value=min_ts.date(),
                    key=f"rule_start_date_{idx}",
                )
                start_time = d1.time_input(
                    "Start time",
                    value=min_ts.time(),
                    key=f"rule_start_time_{idx}",
                )

                end_date = d2.date_input(
                    "End date",
                    value=max_ts.date(),
                    key=f"rule_end_date_{idx}",
                )
                end_time = d2.time_input(
                    "End time",
                    value=max_ts.time(),
                    key=f"rule_end_time_{idx}",
                )

                start_dt = pd.to_datetime(f"{start_date} {start_time}", utc=True)
                end_dt = pd.to_datetime(f"{end_date} {end_time}", utc=True)
            else:
                c5.write("Date range not available")
                start_dt = end_dt = None
        else:
            c5.write("Date range not available")
            start_dt = end_dt = None

        # Save updated rule
        if not remove_rule:
            new_rules.append(
                {
                    "table": selected_table,
                    "column": selected_column,
                    "operator": selected_operator,
                    "value": selected_value,
                    "operation_filter": selected_op_filter,
                    "date_col": real_date_col,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                }
            )

        st.markdown("---")

    # Finalize rules
    rules = new_rules
    st.session_state.analytics_rules = rules

    # AND / OR between rules
    if len(rules) > 1:
        combine_mode = st.radio(
            "Combine rules with:",
            ["AND", "OR"],
            horizontal=True,
            key="rules_combine_mode",
        )
    else:
        combine_mode = "AND"

    compute = st.button("Compute")
    if not compute:
        st.stop()

    # ---------- APPLY RULES ----------
    from collections import defaultdict

    rules_by_table = defaultdict(list)
    for r in rules:
        rules_by_table[r["table"]].append(r)

    results_by_table = {}
    total_rows = 0

    for table_name, tbl_rules in rules_by_table.items():
        df_tbl = get_table_df(table_name)
        if df_tbl.empty:
            continue

        combined_mask = None

        for r in tbl_rules:
            mask = pd.Series([True] * len(df_tbl))

            col = r["column"]
            op = r["operator"]
            val = r["value"]
            op_filter = r["operation_filter"]
            date_col = r["date_col"]
            start_dt = r["start_dt"]
            end_dt = r["end_dt"]

            # ---- Column Filter ----
            if col in df_tbl.columns and op != "All":
                series = df_tbl[col].astype(str)

                if op in ["equals", "not equals", "contains"] and val is not None:
                    val_str = str(val)

                    if op == "equals":
                        mask &= series == val_str
                    elif op == "not equals":
                        mask &= series != val_str
                    elif op == "contains":
                        mask &= series.str.contains(val_str, case=False, na=False)

            # ---- Operation filter ----
            if "operation" in df_tbl.columns and op_filter != "All":
                mask &= df_tbl["operation"].astype(str) == op_filter

            # ---- Date range (timezone-safe) ----
            if date_col and date_col in df_tbl.columns and start_dt and end_dt:
                mask &= (df_tbl[date_col] >= start_dt) & (df_tbl[date_col] <= end_dt)

            # ---- Combine masks ----
            if combined_mask is None:
                combined_mask = mask
            else:
                if combine_mode == "AND":
                    combined_mask &= mask
                else:
                    combined_mask |= mask

        filtered_df = df_tbl[combined_mask] if combined_mask is not None else df_tbl

        results_by_table[table_name] = filtered_df
        total_rows += len(filtered_df)

    # ---------- DISPLAY ----------
    st.markdown("### Results")
    st.write(f"Total rows across all tables: **{total_rows}**")

    for table_name, df_res in results_by_table.items():
        st.subheader(f"📄 {table_name} ({len(df_res)} rows)")
        if df_res.empty:
            st.info("No rows for this table.")
            continue
        st.dataframe(df_res, use_container_width=True)

    st.stop()



#                           DASHBOARD
elif page == "Dashboard":

    st.set_page_config(page_title="Supabase Dashboard", layout="wide")
    st.title("📊 Supabase Dashboard")
    st.sidebar.header("🔍 Select Tables")
    tables = list_tables()
    selected_tables = st.sidebar.multiselect("Choose tables", tables, default=[])
    st.markdown("### 🧪 SQL Query Console")
    sql_input = st.text_area("Write SQL here", placeholder="SELECT * FROM users LIMIT 10;")
    if st.button("Run Query"):
        with st.spinner("Running..."):
            data = run_custom_sql(sql_input)
        if data:
            st.dataframe(pd.DataFrame(data), width='stretch')
        else:
            st.warning("No results")
    st.markdown("---")
    if not selected_tables:
        st.info("Select tables from sidebar.")
    else:
        for table_name in selected_tables:
            st.subheader(f"📄 {table_name}")
            rows = fetch_table(table_name)
            if not rows:
                st.warning("No data")
                continue
            df = pd.DataFrame(rows)
            with st.expander(f"Expand {table_name}", expanded=True):
                view_mode = st.radio(
                    f"View mode for {table_name}",
                    ("Table View", "JSON View"),
                    key=f"mode_{table_name}",
                    horizontal=True,
                )
                if view_mode == "JSON View":
                    st.json(rows)
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect(
                    f"Select columns to display for {table_name}",
                    all_columns,
                    default=all_columns
                )

                st.dataframe(df[selected_columns], use_container_width=True)
