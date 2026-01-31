# Image de base Python 3.13
FROM python:3.13-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source
COPY . .

# Créer les répertoires nécessaires s'ils n'existent pas
RUN mkdir -p /app/mlruns /app/models

# Exposer le port 8000
EXPOSE 8000

# Variable d'environnement pour Python (évite les problèmes de buffering)
ENV PYTHONUNBUFFERED=1

# Variable d'environnement pour Python (évite les problèmes de buffering)
ENV PYTHONUNBUFFERED=1

# ⭐ NOUVEAU : Désactiver IPv6 ⭐
RUN echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]