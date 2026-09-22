import os
import logging
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

bearer_scheme = HTTPBearer(auto_error=False)

_supabase_client = None


def get_supabase():
    global _supabase_client
    if _supabase_client is None:
        try:
            from supabase import create_client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        except Exception as exc:
            log.error("Could not create Supabase client: %s", exc)
            raise RuntimeError("Supabase not configured") from exc
    return _supabase_client


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header required")

    token = credentials.credentials
    try:
        supabase = get_supabase()
        response = supabase.auth.get_user(token)
        user = response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"patient_id": str(user.id), "email": user.email or ""}
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Token validation failed: %s", exc)
        raise HTTPException(status_code=401, detail="Unauthorized")
