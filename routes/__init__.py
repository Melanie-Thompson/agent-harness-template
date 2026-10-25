import importlib
import pkgutil

from fastapi import APIRouter

router = APIRouter()

for module_info in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_info.name}")
    if hasattr(module, "router"):
        router.include_router(module.router)