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

    # ---------------------------------------------------
    # Load Supabase table list
    # ---------------------------------------------------
    tables = list_tables()
    if not tables:
        st.error("No tables available.")
        st.stop()

    # ---------------------------------------------------
    # Load table into DataFrame (cached)
    # ---------------------------------------------------
    @st.cache_data
    def get_table_df(table_name: str) -> pd.DataFrame:
        rows = fetch_table(table_name)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)

        # Convert common date-like columns to datetime
        for col in df.columns:
            if any(k in col.lower() for k in ["updated", "created", "timestamp", "date"]):
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        return df

    # ---------------------------------------------------
    # Utility: find likely matching column name
    # ---------------------------------------------------
    def pick_column(candidates: list[str], cols: list[str]):
        cset = {c.lower(): c for c in cols}
        for c in candidates:
            if c.lower() in cset:
                return cset[c.lower()]
        return None

    # ---------------------------------------------------
    # RULE STATE
    # ---------------------------------------------------
    if "analytics_rules" not in st.session_state:
        st.session_state.analytics_rules = [
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
        ]

    rules = st.session_state.analytics_rules

    # Rule add/remove helpers
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

    st.markdown("### 🔍 Rules")

    # Add rule button
    add_btn_col, _ = st.columns([0.25, 0.75])
    if add_btn_col.button("➕ Add Rule"):
        add_rule()

    new_rules = []

    # ---------------------------------------------------
    # Render Each Rule
    # ---------------------------------------------------
    for idx, rule in enumerate(rules):
        st.markdown(f"#### Rule {idx + 1}")
        c_table, c_remove = st.columns([0.9, 0.1])

        selected_table = c_table.selectbox(
            "Table",
            tables,
            index=tables.index(rule["table"]),
            key=f"table_{idx}"
        )

        remove_clicked = c_remove.button("❌", key=f"remove_{idx}")

        df_tbl = get_table_df(selected_table)
        cols = df_tbl.columns.tolist()

        if df_tbl.empty:
            st.warning(f"No data in `{selected_table}`.")
            if not remove_clicked:
                new_rules.append(rule | {"table": selected_table})
            st.markdown("---")
            continue

        # Column selector
        col1, col2, col3 = st.columns([2, 2, 3])

        selected_column = col1.selectbox(
            "Column",
            cols,
            index=cols.index(rule["column"]) if rule["column"] in cols else 0,
            key=f"col_{idx}"
        )

        operators = ["All", "equals", "not equals", "contains"]
        selected_operator = col2.selectbox(
            "Operator",
            operators,
            index=operators.index(rule["operator"]),
            key=f"op_{idx}"
        )

        # Value dropdown (auto derived from table)
        values = df_tbl[selected_column].dropna().astype(str).unique().tolist()
        values.sort()

        if selected_operator != "All" and values:
            selected_value = col3.selectbox(
                "Value",
                values,
                index=values.index(rule["value"]) if rule["value"] in values else 0,
                key=f"val_{idx}"
            )
        else:
            col3.write("All values")
            selected_value = None

        # Operation column detection
        operation_col = pick_column(["operation"], cols)

        c_op, c_date = st.columns([1.2, 3])

        # Operation filter dropdown
        if operation_col:
            op_vals = df_tbl[operation_col].dropna().astype(str).unique().tolist()
            op_vals.sort()

            selected_operation = c_op.selectbox(
                "Operation",
                ["All"] + op_vals,
                index=(["All"] + op_vals).index(rule["operation_filter"])
                if rule["operation_filter"] in (["All"] + op_vals) else 0,
                key=f"op_filter_{idx}"
            )
        else:
            selected_operation = "All"
            c_op.write("Operation column not found")

        # DATE FILTERS
        date_col = pick_column(
            ["updatedat", "updated_at", "createdat", "created_at", "timestamp"],
            cols
        )

        if date_col:
            # Convert column to datetime
            df_tbl[date_col] = pd.to_datetime(df_tbl[date_col], errors="coerce", utc=True)

            # --- NEW DEFAULT RANGE ---
            now = pd.Timestamp.now(tz="UTC")
            default_start = now - pd.DateOffset(months=2)
            default_end = now + pd.DateOffset(days=1)

            d1, d2 = c_date.columns(2)

            # Default start values
            start_date = d1.date_input(
                "Start Date",
                value=default_start.date(),
                key=f"sd_{idx}"
            )
            start_time = d1.time_input(
                "Start Time",
                value=default_start.time(),
                key=f"st_{idx}"
            )

            # Default end values
            end_date = d2.date_input(
                "End Date",
                value=default_end.date(),
                key=f"ed_{idx}"
            )
            end_time = d2.time_input(
                "End Time",
                value=default_end.time(),
                key=f"et_{idx}"
            )

            # Convert to timezone-aware timestamps
            start_dt = pd.to_datetime(f"{start_date} {start_time}", utc=True)
            end_dt = pd.to_datetime(f"{end_date} {end_time}", utc=True)
        else:
            c_date.write("No date column found")
            start_dt = end_dt = None


        if not remove_clicked:
            new_rules.append({
                "table": selected_table,
                "column": selected_column,
                "operator": selected_operator,
                "value": selected_value,
                "operation_filter": selected_operation,
                "date_col": date_col,
                "start_dt": start_dt,
                "end_dt": end_dt,
            })

        st.markdown("---")

    # Update state
    st.session_state.analytics_rules = new_rules
    rules = new_rules

    # ---------------------------------------------------
    # AND / OR COMBINATION
    # ---------------------------------------------------
    if len(rules) > 1:
        combine_mode = st.radio(
            "Combine rules with:",
            ["AND", "OR"],
            horizontal=True,
            key="combine_mode"
        )
    else:
        combine_mode = "AND"

    # ---------------------------------------------------
    # COMPUTE BUTTON
    # ---------------------------------------------------
    compute = st.button("Compute")
    if not compute:
        st.stop()

    # ---------------------------------------------------
    # APPLY RULES ENGINE (FIXED VERSION)
    # ---------------------------------------------------
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

        rule_masks = []

        for r in tbl_rules:
            mask = pd.Series([True] * len(df_tbl))

            col = r["column"]
            op = r["operator"]
            val = r["value"]
            op_filter = r["operation_filter"]
            date_col = r["date_col"]
            start_dt = r["start_dt"]
            end_dt = r["end_dt"]

            # Column filter
            if op != "All" and col in df_tbl.columns and val is not None:
                series = df_tbl[col].astype(str)
                if op == "equals":
                    mask &= series == val
                elif op == "not equals":
                    mask &= series != val
                elif op == "contains":
                    mask &= series.str.contains(val, case=False, na=False)

            # Operation filter
            if op_filter != "All" and "operation" in df_tbl.columns:
                mask &= df_tbl["operation"].astype(str) == op_filter

            # Date filter
            if date_col and date_col in df_tbl.columns and start_dt and end_dt:
                mask &= (df_tbl[date_col] >= start_dt) & (df_tbl[date_col] <= end_dt)

            rule_masks.append(mask)

        # Combine rule masks correctly
        if len(rule_masks) == 0:
            combined_mask = pd.Series([True] * len(df_tbl))
        elif len(rule_masks) == 1:
            combined_mask = rule_masks[0]
        else:
            if combine_mode == "AND":
                combined_mask = rule_masks[0]
                for m in rule_masks[1:]:
                    combined_mask &= m
            else:
                combined_mask = rule_masks[0]
                for m in rule_masks[1:]:
                    combined_mask |= m

        filtered_df = df_tbl[combined_mask]
        results_by_table[table_name] = filtered_df
        total_rows += len(filtered_df)

    # ---------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------
    st.markdown("### 📊 Results")
    st.write(f"Total rows returned: **{total_rows}**")

    for table_name, df in results_by_table.items():
        st.subheader(f"{table_name} — {len(df)} rows")
        if df.empty:
            st.info("No matching rows.")
        else:
            st.dataframe(df, use_container_width=True)

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
