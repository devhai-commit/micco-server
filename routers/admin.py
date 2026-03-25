from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload
from typing import Optional

from database import get_db
from models import User, Department
from schemas import DepartmentResponse
from auth import get_current_user, hash_password

router = APIRouter(prefix="/api/admin", tags=["Admin"])


def _require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "Admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ═══════════════════════════════════════════════════════════════
# DEPARTMENTS
# ═══════════════════════════════════════════════════════════════

@router.get("/departments")
def list_departments(
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    depts = db.query(Department).order_by(Department.name).all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "description": d.description,
            "created_at": d.created_at,
            "user_count": len(d.users),
        }
        for d in depts
    ]


@router.post("/departments", status_code=201)
def create_department(
    body: dict,
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Tên phòng ban không được trống")
    existing = db.query(Department).filter(Department.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Phòng ban đã tồn tại")
    dept = Department(name=name, description=body.get("description", ""))
    db.add(dept)
    db.commit()
    db.refresh(dept)
    return {"id": dept.id, "name": dept.name, "description": dept.description, "created_at": dept.created_at, "user_count": 0}


@router.put("/departments/{dept_id}")
def update_department(
    dept_id: int,
    body: dict,
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    dept = db.query(Department).filter(Department.id == dept_id).first()
    if not dept:
        raise HTTPException(status_code=404, detail="Phòng ban không tồn tại")
    if "name" in body:
        new_name = (body["name"] or "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Tên phòng ban không được trống")
        dup = db.query(Department).filter(Department.name == new_name, Department.id != dept_id).first()
        if dup:
            raise HTTPException(status_code=400, detail="Tên phòng ban đã tồn tại")
        dept.name = new_name
    if "description" in body:
        dept.description = body["description"]
    db.commit()
    db.refresh(dept)
    return {"id": dept.id, "name": dept.name, "description": dept.description, "created_at": dept.created_at, "user_count": len(dept.users)}


@router.delete("/departments/{dept_id}", status_code=204)
def delete_department(
    dept_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    dept = db.query(Department).filter(Department.id == dept_id).first()
    if not dept:
        raise HTTPException(status_code=404, detail="Phòng ban không tồn tại")
    db.delete(dept)
    db.commit()
    return None


# ═══════════════════════════════════════════════════════════════
# USERS
# ═══════════════════════════════════════════════════════════════

@router.get("/stats")
def admin_stats(
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    total_users = db.query(User).count()
    from models import Document
    from sqlalchemy import func
    storage = db.query(func.coalesce(func.sum(Document.size_bytes), 0)).scalar()
    def fmt(b):
        if b >= 1 << 30:
            return f"{b / (1 << 30):.1f} GB"
        return f"{b / (1 << 20):.1f} MB"
    return {
        "totalUsers": total_users,
        "storageUsed": fmt(storage),
        "activeSessions": 0,
        "totalUsersChange": "",
        "storageChange": "",
        "activeSessionsChange": "",
    }


@router.get("/users")
def list_users(
    search: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    query = db.query(User).options(joinedload(User.department))
    if search:
        query = query.filter(
            User.name.ilike(f"%{search}%") | User.email.ilike(f"%{search}%")
        )
    if role and role != "All":
        query = query.filter(User.role == role)
    total = query.count()
    users = query.order_by(User.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {
        "users": [
            {
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "role": u.role,
                "department_id": u.department_id,
                "department_name": u.department.name if u.department else None,
                "avatar": u.avatar,
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in users
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("/users", status_code=201)
def create_user(
    body: dict,
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    email = (body.get("email") or "").strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email không được trống")
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email đã tồn tại")
    password = body.get("password") or "123456"
    user = User(
        name=body.get("name", ""),
        email=email,
        hashed_password=hash_password(password),
        role=body.get("role", "Nhân viên"),
        department_id=body.get("department_id"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role}


@router.put("/users/{user_id}")
def update_user(
    user_id: int,
    body: dict,
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    if "name" in body:
        user.name = body["name"]
    if "email" in body:
        dup = db.query(User).filter(User.email == body["email"], User.id != user_id).first()
        if dup:
            raise HTTPException(status_code=400, detail="Email đã tồn tại")
        user.email = body["email"]
    if "role" in body:
        user.role = body["role"]
    if "department_id" in body:
        user.department_id = body["department_id"]
    if body.get("password"):
        user.hashed_password = hash_password(body["password"])
    db.commit()
    db.refresh(user)
    return {"id": user.id, "name": user.name, "email": user.email, "role": user.role}


@router.delete("/users/{user_id}", status_code=204)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Người dùng không tồn tại")
    db.delete(user)
    db.commit()
    return None


# ═══════════════════════════════════════════════════════════════
# GRAPHRAG COMMUNITIES
# ═══════════════════════════════════════════════════════════════

@router.post("/communities/build")
def build_communities_endpoint(
    db: Session = Depends(get_db),
    _admin: User = Depends(_require_admin),
):
    """Trigger Leiden community detection + LLM summary generation.

    This is an expensive batch operation — run it after bulk document ingest,
    not on every upload.
    """
    from services.community_service import build_communities

    result = build_communities(db)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result
