"""
Camera CRUD routes.

Endpoints:
    POST   /cameras/           – register a new camera
    GET    /cameras/           – list all cameras
    GET    /cameras/{id}       – get a single camera
    PATCH  /cameras/{id}       – update camera fields
    DELETE /cameras/{id}       – delete a camera
"""

from fastapi import APIRouter, HTTPException, status

from src.api.deps import DbDep
from src.api.schemas import CameraCreate, CameraOut, CameraUpdate
from src.database import crud

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.post("/", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(body: CameraCreate, db: DbDep):
    """Register a new camera / scan location."""
    return crud.create_camera(
        db,
        name=body.name,
        lat=body.lat,
        lon=body.lon,
        roi_coords=body.roi_coords,
        is_active=body.is_active,
    )


@router.get("/", response_model=list[CameraOut])
def list_cameras(db: DbDep, skip: int = 0, limit: int = 100):
    """Return all registered cameras."""
    return crud.list_cameras(db, skip=skip, limit=limit)


@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: int, db: DbDep):
    """Return a single camera by ID."""
    cam = crud.get_camera(db, camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: int, body: CameraUpdate, db: DbDep):
    """Update camera metadata."""
    cam = crud.update_camera(db, camera_id, **body.model_dump(exclude_none=True))
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return cam


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: int, db: DbDep):
    """Delete a camera and all associated scans."""
    deleted = crud.delete_camera(db, camera_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Camera not found")
