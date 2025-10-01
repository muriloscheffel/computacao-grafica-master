# Estou utilizando venv do python para rodar meu ambiente
# source .venv/bin/activate

# (se tiver venv)
source .venv/bin/activate

# deps mínimos
pip install "Pillow>=10,<11" "numpy>=1.24,<3" PyOpenGL==3.1.7

# se quiser garantir a plataforma
export PYOPENGL_PLATFORM=glut

# executar
python3 main.py /mnt/c/projects/comput-grafica/t1/data
# ou apenas:
python3 main.py
