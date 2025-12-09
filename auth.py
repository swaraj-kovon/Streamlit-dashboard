from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ============================================================
# Signup (email verification required)
# ============================================================
def signup(email: str, password: str):
    try:
        res = supabase.auth.sign_up(
            {
                "email": email,
                "password": password,
            }
        )

        if res.user is None:
            return {"error": "Signup failed or email already registered"}

        return {"user": res.user.email}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Sign-in
# ============================================================
def signin(email: str, password: str):
    try:
        res = supabase.auth.sign_in_with_password(
            {
                "email": email,
                "password": password
            }
        )

        if res.session is None:
            return {"error": "Invalid credentials or email not verified"}

        return {"session": res.session, "user": res.user.email}

    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Who is currently logged in?
# ============================================================
def get_current_user():
    try:
        session = supabase.auth.get_session()
        if session and session.session and session.session.user:
            return session
        return None
    except Exception:
        return None


# ============================================================
# Logout
# ============================================================
def signout():
    try:
        supabase.auth.sign_out()
        return True
    except Exception:
        return False
