import subprocess

with open("version.txt") as file:
    version = file.read()
    subprocess.run([
        "nuitka.cmd"
        , "src/main.py"
        , f"--windows-product-name=Note"
        # , "--windows-company-name=CNSCAN"
        , "--standalone"
        , "--plugin-enable=pyside6"
        , "--nofollow-import-to=QtMultimedia"
        , "--nofollow-import-to=QtWebEngine"
        , "--windows-console-mode=disable"
        , "--include-package=qt_material"
        , f"--windows-icon-from-ico=./image/book_ribbon.png"
        , "--include-data-dir=./image=image"
        , "--include-data-dir=./style=style"
        , "--include-data-file=./version.txt=version.txt"
        , "--include-data-file=./dark_theme.xml=dark_theme.xml"
        , "--include-data-file=./light_theme.xml=light_theme.xml"
        , f"--output-filename=Note"
        , "--output-dir=./dist"
        , f"--windows-file-version={version}"
    ])
