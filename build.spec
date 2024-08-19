import os
import platform
import gooey
gooey_root = os.path.dirname(gooey.__file__)
gooey_languages = Tree(os.path.join(gooey_root, 'languages'), prefix = 'gooey/languages')
gooey_images = Tree(os.path.join(gooey_root, 'images'), prefix = 'gooey/images')

from PyInstaller.building.api import EXE, PYZ, COLLECT
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.datastruct import Tree
from PyInstaller.building.osx import BUNDLE

block_cipher = None

a = Analysis(['/Users/adityasriram/Documents/htp-screening/src/main.py'],  # replace me with your path
             pathex=['/Users/adityasriram/Documents/htp-screening/src'],
             binaries=None,
             data=None,
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None,
             excludes=None
             )
pyz = PYZ(a.pure)

options = [('u', None, 'OPTION')]

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          # gooey_languages,
          # gooey_images,
          name='DMREF BARCODE',
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon=os.path.join(gooey_root, 'images', 'program_icon.ico'))

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='DMREF BARCODE')


info_plist = {'addition_prop': 'additional_value'}
app = BUNDLE(exe,
             name='DMREF BARCODE.app',
             bundle_identifier=None,
             info_plist=info_plist
            )
