"""Select existing immutable LIBERO assets without changing the installed package."""
from pathlib import Path
import hashlib,os,sys
ROOT=Path(__file__).resolve().parents[3]
ASSETS=Path("/mnt/cpfs/zbl-cpfs-new/dataset/leon/libero/assets/90001343cb134b7e26e18fde0fa2416f3ed6e6a3")
def configure_assets():
    config=ROOT/"experiments/r16_p14_libero_stage1/libero_config"
    scene=ASSETS/"scenes/libero_tabletop_base_style.xml"
    if not scene.is_file() or not (config/"config.yaml").is_file():
        raise RuntimeError("exact frozen LIBERO assets/config unavailable")
    os.environ["LIBERO_CONFIG_PATH"]=str(config)
    sys.path.insert(0,str(ROOT))
    # Bind the existing lookup cache before env imports; never edit shared source.
    import libero.libero as package
    expected=ROOT/"libero/libero/__init__.py"
    if Path(package.__file__).resolve()!=expected.resolve():
        raise RuntimeError("wrong LIBERO package; repository source required")
    package.libero_config_path=str(config)
    package.config_file=str(config/"config.yaml")
    package._assets_path_cache=str(ASSETS)
    if Path(package.get_assets_path()).resolve()!=ASSETS.resolve():
        raise RuntimeError("LIBERO asset selection drift")
    return {"assets":str(ASSETS),"scene_sha256":hashlib.sha256(scene.read_bytes()).hexdigest(),
            "config_sha256":hashlib.sha256((config/"config.yaml").read_bytes()).hexdigest(),
            "package_source":str(Path(package.__file__).resolve()),
            "package_sha256":hashlib.sha256(Path(package.__file__).read_bytes()).hexdigest()}
