from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from database import get_db
from models import User

# ─── Password Hashing ───────────────────────────────────────
# Uses bcrypt directly (passlib 1.7.4 is incompatible with bcrypt ≥ 4.x).
# Passwords are truncated to 72 bytes before hashing — bcrypt's hard limit.
# Hashes produced are identical $2b$ format; existing stored hashes are unaffected.

_BCRYPT_MAX = 72


def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode()[:_BCRYPT_MAX], _bcrypt.gensalt()).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return _bcrypt.checkpw(plain_password.encode()[:_BCRYPT_MAX], hashed_password.encode())


# ─── JWT Token ───────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    # JWT RFC 7519 requires "sub" to be a string
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─── FastAPI Dependency ─────────────────────────────────────

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    payload = decode_access_token(credentials.credentials)
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Convert back from string to int (JWT sub must be a string)
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user
