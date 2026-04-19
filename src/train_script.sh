#!/bin/bash
#SBATCH --nodelist=gpue07        # Nœud spécifique
#SBATCH --cpus-per-task=16        # Nombre de CPU
#SBATCH --mem=32G                # Mémoire totale
#SBATCH -J whisper-finetune      # Nom du job
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --time=12:00:00          # Temps max (12h)
#SBATCH --output=whisper_%j.out  # Fichier de sortie
#SBATCH --error=whisper_%j.err   # Fichier d'erreur
#SBATCH --mail-type=ALL          # Notifications par email
#SBATCH --mail-user=Ali.Dinar-Bous.Etu@univ-lemans.fr


# Lancer le script Python
python train_kurd_to_zazaki.py

