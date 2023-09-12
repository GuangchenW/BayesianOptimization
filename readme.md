#How to build
1. In the project root directory, run ```python -m venv``` to initialize the virtual environment.
2. Run ```source venv/bin/activate``` for mac, ```Scripts\Activate.ps1``` for windows to activate the venv.
3. Run ```git submodule init``` and then ```git submodule update``` to initialize the repo submodules.
4. Run ```pipenv install``` to install all dependencies from the pipfile.
5. Run ```pipenv install -e GPyOpt``` to install the local package.