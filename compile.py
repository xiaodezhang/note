import subprocess
from src.config import Config
import shutil

config = Config()
app_name = config.name
file_name = config.config_file
icon = config.icon
with open("version.txt") as file:
    version = file.read()
    subprocess.run([
        "nuitka.cmd"
        , "src/main.py"
        , f"--windows-product-name={app_name}"
        , "--windows-company-name=CNSCAN"
        , "--standalone"
        , "--plugin-enable=pyside6"
        , "--nofollow-import-to=QtMultimedia"
        , "--nofollow-import-to=QtWebEngine"
        , "--windows-console-mode=disable"
        , "--include-package=qt_material"
        , f"--windows-icon-from-ico=./image/{icon}"
        , "--include-data-dir=./image=image"
        , "--include-data-dir=./style=style"
        , "--include-data-dir=./translations=translations"
        , "--include-data-dir=./net=net"
        , "--include-data-file=./version.txt=version.txt"
        , "--include-data-file=./dark_theme.xml=dark_theme.xml"
        , "--include-data-file=./light_theme.xml=light_theme.xml"
        , f"--include-data-file=./{file_name}={file_name}"
        , f"--output-filename={app_name}"
        , "--output-dir=./dist"
        , f"--windows-file-version={version}"
    ])


src = "driver"           # 源目录
dst = "dist/main.dist/driver"      # 目标目录

shutil.copytree(src, dst, dirs_exist_ok=True)
