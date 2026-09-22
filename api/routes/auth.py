import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_supabase, get_current_user

log = logging.getLogger(__name__)
router = APIRouter()


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/signup")
async def signup(req: SignupRequest):
    """
    Create a new Supabase Auth user.
    The patient profile is created separately via POST /patients/me.
    Returns access_token for immediate use.
    """
    try:
        supabase = get_supabase()
        resp = supabase.auth.sign_up({"email": req.email, "password": req.password})
        if not resp.user:
            raise HTTPException(status_code=400, detail="Signup failed — no user returned")
        return {
            "user_id": str(resp.user.id),
            "email": resp.user.email,
            "access_token": resp.session.access_token if resp.session else None,
            "refresh_token": resp.session.refresh_token if resp.session else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Signup error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/login")
async def login(req: LoginRequest):
    """Sign in with email + password. Returns Supabase JWT."""
    try:
        supabase = get_supabase()
        resp = supabase.auth.sign_in_with_password({"email": req.email, "password": req.password})
        if not resp.user or not resp.session:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {
            "access_token": resp.session.access_token,
            "refresh_token": resp.session.refresh_token,
            "user_id": str(resp.user.id),
            "email": resp.user.email,
            "token_type": "bearer",
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Login error: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    try:
        supabase = get_supabase()
        supabase.auth.sign_out()
    except Exception:
        pass
    return {"message": "Logged out successfully"}


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    """Return the authenticated user's ID and email."""
    return current_user
