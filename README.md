<div align="center">

<img src="https://img.shields.io/badge/WER%20final-43%25-4CAF50?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Whisper-Fine--tuning-FF6F00?style=for-the-badge&logo=openai&logoColor=white"/>
<img src="https://img.shields.io/badge/Langue-Zazaki-8E24AA?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Transfer%20Learning-Kurde%20Nord-1565C0?style=for-the-badge&logoColor=white"/>
<img src="https://img.shields.io/badge/Pseudo--labeling-Augmentation-00897B?style=for-the-badge&logoColor=white"/>

<br/><br/>

# Transcription automatique des langues peu dotées
### Fine-tuning de Whisper pour le Zazaki via transfert linguistique progressif

<br/>

> Le zazaki est parlé par ~3 millions de locuteurs mais dispose de **moins d'une heure** de données audio transcrites.  
> Ce projet démontre qu'une stratégie de transfert par langue intermédiaire et l'augmentation des données par apprentissage semi-supervisé permettent de réduire le WER de **111,67 % à 43 %**, sans collecter une seule nouvelle donnée annotée manuellement.

</div>

---

## Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Langue cible : le Zazaki](#langue-cible--le-zazaki)
- [Architecture du système ASR](#architecture-du-système-asr)
- [Jeux de données](#jeux-de-données)
- [Métriques d'évaluation](#métriques-dévaluation)
- [Pipeline expérimental](#pipeline-expérimental)
- [Expériences & Résultats](#expériences--résultats)
- [Récapitulatif des performances](#récapitulatif-des-performances)
- [Perspectives](#perspectives)

---

## Vue d'ensemble

Ce projet explore la transcription automatique du **zazaki**, une langue à très faibles ressources numériques, en exploitant des stratégies de fine-tuning du modèle **Whisper** d'OpenAI. L'hypothèse centrale : il est possible d'adapter un système ASR performant à une langue quasi absente des corpus d'entraînement existants, en tirant parti de la **proximité linguistique** avec des langues mieux dotées.

Le travail s'articule en **6 expériences progressives**, de la validation du pipeline jusqu'à l'augmentation de données par pseudo-labeling, avec comme fil conducteur la stratégie de **transfert par langue intermédiaire** (Turc → Kurde du Nord → Zazaki).

**Durée :** octobre 2025 – mars 2026 (9 semaines de travail effectif)  
---

## Langue cible : le Zazaki

Le zazaki est une langue minoritaire parlée principalement en Turquie, classée **langue en danger** par l'UNESCO. Sa rareté numérique en fait un cas d'étude idéal pour les approches low-resource.

```
Famille      : Indo-européenne, branche iranienne nord-ouest
Locuteurs    : ~3 millions
Alphabet     : Latin (proche du turc)
Statut       : Langue en danger (UNESCO)
Contrainte   : < 1 heure de données audio transcrites disponibles
```

La parenté linguistique avec le **kurde du Nord** (même branche iranienne nord-ouest) et l'influence historique du **turc** (alphabet partagé, lexique commun) sont les deux leviers exploités dans ce projet.

---

## Architecture du système ASR

Le projet utilise **Whisper** (OpenAI), un modèle Transformer encodeur-décodeur pré-entraîné sur 680 000 heures d'audio multilingue.

<div align="center">
  <img src="assets/screenshots/architecture whisper.PNG" width="800"/>
</div>

```
Signal audio brut (48 kHz)
        │
        ▼
1. Prétraitement
   |
   ├── Rééchantillonnage à 16 kHz
   └── Extraction spectrogramme log-Mel
        │
        ▼
2. Encodeur Transformer
   └── Représentation latente du signal
       (phonétique + prosodie + bruit + accent)
        │
        ▼
3. Décodeur Transformer (autorégressif)
   └── Génération token par token via cross-attention
        │
        ▼
   Transcription textuelle
```

**Pourquoi Whisper pour le low-resource ?**
Son pré-entraînement massif multilingue lui confère des représentations acoustiques généralisables. Son architecture encodeur-décodeur permet un transfert efficace vers de nouvelles langues avec peu de données via le fine-tuning.

---

## Jeux de données

### Données de validation (langues de référence)

| Corpus | Langues utilisées | Usage |
|--------|------------------|-------|
| FLEURS | Anglais, Français | Validation du pipeline (Expérience 1) |
| Common Voice (Mozilla) | Turc, Kurde du Nord | Sélection langue source + entraînement pivot |

### Données zazaki

| Partition | Proportion | Échantillons |
|-----------|-----------|--------------|
| Entraînement | 47,5 % | ~813 |
| Validation | 27 % | ~462 |
| Test | 25,5 % | ~437 |
| **Total** | **100 %** | **~1 712** |

Le corpus zazaki, issu d'une dizaine de locuteurs différents, représente moins d'une heure d'audio transcrit contrainte centrale de tout le projet.

---

## Métriques d'évaluation

### WER — Taux d'erreur de mots

Métrique principale pour évaluer un système ASR. Elle mesure la proportion de mots mal transcrits par rapport à la référence.

```
WER = (S + D + I) / N × 100

S = substitutions   (mot remplacé par un autre)
D = suppressions    (mot omis)
I = insertions      (mot ajouté)
N = nombre total de mots dans la transcription de référence
```

Un WER de 0 % signifie une transcription parfaite. Un WER > 100 % indique que le modèle insère plus de mots qu'il n'en transcrit correctement (cas du zero-shot sur une langue inconnue).

### CER — Taux d'erreur de caractères

Même principe que le WER mais au niveau des caractères. Particulièrement utile pour les langues peu dotées car moins sensible aux découpages de mots non standards.

---

## Pipeline expérimental

Les 6 expériences suivent une progression incrémentale, partant d'un modèle généraliste pour aboutir à un modèle spécialisé sur le zazaki.

```
Exp. 1 — Validation pipeline          Vérifier que l'environnement est fiable
         ↓
Exp. 2 — Zero-shot Zazaki             Mesurer le point de départ, choisir la langue source
         ↓
Exp. 3 — Fine-tuning Turc → Zazaki    Baseline de référence avec transfert direct
         ↓
Exp. 4 — Fine-tuning Turc → Kurde    Construire un modèle pivot iranien nord-ouest
         ↓
Exp. 5 — Fine-tuning Kurde → Zazaki   Exploiter la proximité linguistique réelle
         ↓
Exp. 6 — Pseudo-labeling              Augmenter les données sans annotation manuelle
```

---

## Expériences & Résultats

---

### Expérience 1 — Validation du pipeline d'inférence

**Objectif :** Avant de toucher au zazaki, vérifier que le pipeline d'inférence est techniquement fiable en le testant sur des langues pour lesquelles les benchmarks officiels existent.

**Protocole :** Inférence avec Whisper sur les partitions de test FLEURS (anglais, français) et Common Voice (turc). Normalisation textuelle appliquée avant calcul des métriques (minuscules, suppression de ponctuation).

| Langue | WER (%) | CER (%) |
|--------|---------|---------|
| Anglais | 7,09 | 3,18 |
| Français | 13,36 | 5,52 |
| Turc | 17,54 | 4,64 |

**Analyse :** Les résultats présentent un écart d'environ 2 % par rapport aux benchmarks officiels d'OpenAI, probablement dû aux variations de normalisation textuelle. L'environnement est validé et les résultats sont reproductibles. On peut passer à la suite.

---

### Expérience 2 — Évaluation Zero-shot sur le Zazaki

**Objectif :** Mesurer les performances natives de Whisper sur le zazaki (sans aucune adaptation), et identifier quelle langue source offre le meilleur point de départ pour le fine-tuning.

**Protocole :** Tester Whisper Small et Base, pré-entraînés sur différentes langues (anglais, français, turc), directement sur la partition de test zazaki sans fine-tuning préalable.

| Configuration | Modèle | WER (%) | CER (%) |
|---------------|--------|---------|---------|
| Whisper Anglais → Zazaki | Small | 134,73 | 78,05 |
| Whisper Anglais → Zazaki | Base | 172,65 | 109,42 |
| Whisper Français → Zazaki | Small | 130,93 | 86,73 |
| Whisper Français → Zazaki | Base | 111,67 | 49,94 |
| Whisper Turc → Zazaki | Small | 126,59 | 145,55 |
| **Whisper Turc → Zazaki** | **Base** | **111,67** | **49,94** |

**Analyse :** Les WER > 100 % confirment que le zazaki est absent des données d'entraînement de Whisper. Le modèle turc Base obtient le CER le plus bas (49,94 %), ce qui s'explique par trois facteurs : le zazaki et le turc partagent le même alphabet latin avec des conventions orthographiques proches, le zazaki a subi une forte influence phonétique et lexicale du turc, et Whisper a été pré-entraîné sur davantage de données turques que de données d'autres langues.

**Décision :** Le turc est retenu comme langue source pour toutes les expériences de fine-tuning.

---

### Expérience 3 — Fine-tuning direct : Turc → Zazaki *(baseline)*

**Objectif :** Établir une baseline en fine-tunant directement le modèle Whisper-Base turc sur le corpus zazaki.

**Protocole :**

```
Modèle de départ  : Whisper-Base pré-entraîné sur le turc
Données           : Corpus zazaki (~813 échantillons d'entraînement)
Learning rate     : 1 × 10⁻⁵
Époques           : 5
Batch size        : 8
```

**Résultat :**

| Approche | WER (%) | Gain vs zero-shot |
|----------|---------|-------------------|
| Zero-shot (Exp. 2) | 111,67 % | — |
| **Fine-tuning direct** | **72,94 %** | **−38,73 %** |

**Analyse :** Le fine-tuning réduit le WER de près de 39 %. C'est un gain considérable, mais un WER de 72,94 % reste insuffisant pour une utilisation réelle. Ce score devient la **référence à battre** pour les expériences suivantes. L'intuition est que le turc, bien que partageant l'alphabet et une partie du lexique zazaki, n'en est pas suffisamment proche phonologiquement pour un transfert optimal.

---

### Expérience 4 — Fine-tuning : Turc → Kurde du Nord *(construction du modèle pivot)*

**Objectif :** Construire un modèle intermédiaire expert en kurde du Nord, qui servira de point de départ pour le transfert final vers le zazaki. Le kurde du Nord et le zazaki appartiennent tous deux à la **branche iranienne nord-ouest** des langues indo-européennes , leur proximité phonologique et morphologique est donc bien plus forte que celle entre le turc et le zazaki.

**Protocole :**

```
Modèle de départ  : Whisper-Base pré-entraîné sur le turc
Données           : Corpus Common Voice kurde du Nord (100+ locuteurs)
Hyperparamètres   : identiques à l'Expérience 3
Pas d'entraînement: 750 steps
```

**Résultat :**

| Langue cible | WER (%) |
|--------------|---------|
| Kurde du Nord | **28,11 %** |

**Analyse :** Un WER de 28,11 % sur le kurde confirme que le modèle a bien assimilé les spécificités phonologiques et morphologiques de la branche iranienne nord-ouest. Ce modèle pivot constitue une base bien plus pertinente que le modèle turc pour le transfert final vers le zazaki — il "parle déjà" la même famille de langues.

---

### Expérience 5 — Fine-tuning : Kurde du Nord → Zazaki *(transfert pivot)*

**Objectif :** Exploiter le modèle pivot kurde (Expérience 4) pour fine-tuner vers le zazaki, en pariant sur la proximité linguistique réelle entre les deux langues.

**Protocole :**

```
Modèle de départ  : Whisper-Base fine-tuné sur le kurde du Nord (Exp. 4)
Données           : Corpus zazaki (~813 échantillons d'entraînement)
Learning rate     : 3 × 10⁻⁵  (réduit pour une convergence plus stable)
Hyperparamètres   : schedular cosine, accumulation de graidient: 4 batch de 4 échantillon avec un effectif de 32
```

**Résultat :**

| Approche | WER (%) | Gain vs baseline |
|----------|---------|-----------------|
| Fine-tuning direct Turc → Zazaki (Exp. 3) | 72,94 % | — |
| **Transfert pivot Kurde → Zazaki** | **53,86 %** | **−19,08 %** |

**Récapitulatif de la progression :**

```
Zero-shot          :  WER = 111,67 %   (point de départ)
Fine-tuning direct :  WER =  72,94 %   (−38,73 %)
Transfert pivot    :  WER =  53,86 %   (−19,08 % supplémentaires)
```

**Analyse :** La stratégie de langue pivot apporte un gain net de **19%** par rapport à la baseline directe. Ce résultat valide l'hypothèse centrale du projet : exploiter la proximité linguistique réelle (famille iranienne nord-ouest) est plus efficace qu'un transfert direct depuis une langue influente mais typologiquement distante. Le modèle à 53,86 % est suffisamment stable pour générer des pseudo-labels fiables , ce qui ouvre la voie à l'Expérience 6.

---

### Expérience 6 — Augmentation par pseudo-labeling

**Objectif :** Pallier la pénurie de données annotées en utilisant le meilleur modèle obtenu (WER 53,86 %) pour transcrire automatiquement des fichiers audio zazaki bruts, non supervisés. Ces transcriptions automatique, appelées **pseudo-labels** sont réinjectées dans l'entraînement après filtrage qualité.

**Méthodologie (processus itératif en 3 étapes) :**

```
Données audio brutes (non supervisées)
        │
        ▼
Étape 1 — Inférence
  Le modèle (WER 53,86 %) génère des transcriptions automatiques
  sur tous les fichiers audio bruts disponibles
  → 25 095 segments générés
        │
        ▼
Étape 2 — Filtrage par score de confiance
  Seules les transcriptions avec un score de confiance > seuil sont conservées
  → 6 716 segments retenus (26,7 % du total)
        │
        ▼
Étape 3 — Ré-entraînement
  Le modèle est ré-entraîné sur :
  données originales + 6 716 pseudo-labels filtrés
  → WER = 43 %
```

**Résultats du processus :**

| Étape | Description | Résultat |
|-------|-------------|----------|
| 1 | Segmentation et inférence sur audio brut | 25 095 segments générés |
| 2 | Filtrage par score de confiance | 6 716 segments conservés (26,7 %) |
| 3 | Ré-entraînement du modèle final | **WER = 43 %** |

**Progression globale du projet :**

```
Expérience 2  — Zero-shot            :  WER = 111,67 %
Expérience 3  — Fine-tuning direct   :  WER =  72,94 %   (−38,73 %)
Expérience 5  — Transfert pivot      :  WER =  53,86 %   (−19,08 %)
Expérience 6  — Pseudo-labeling      :  WER =  43,00 %   (−10,86 %)
─────────────────────────────────────────────────────────
Gain total                           :              −68,67 pts  (−61,5 %)
```

**Analyse :** Le filtrage sévère (conservation de seulement 26,7 % des pseudo-labels) est une décision volontaire : mieux vaut un corpus plus petit mais propre qu'un grand corpus bruité qui ferait mémoriser ses propres erreurs au modèle. La réduction de 10,86% valide l'efficacité du pseudo-labeling pour les langues peu dotées, même avec un modèle de départ imparfait.

---

## Récapitulatif des performances

| Expérience | Stratégie | WER (%) | Gain cumulé |
|------------|-----------|---------|-------------|
| Exp. 2 | Zero-shot (point de départ) | 111,67 | — |
| Exp. 3 | Fine-tuning Turc → Zazaki | 72,94 | −38,73 % |
| Exp. 5 | Fine-tuning Kurde → Zazaki (pivot) | 53,86 | −57,81 % |
| Exp. 6 | Pseudo-labeling | **43,00** | **−68,67 %** |

La stratégie de **transfert progressif par langue intermédiaire ** combinée au **pseudo-labeling** permet de diviser le taux d'erreur par plus de 2,5 en partant d'un corpus de moins d'une heure.

> Un WER de 43 % reste insuffisant pour un déploiement opérationnel (seuil industriel : ~15–20 %), mais constitue un résultat solide compte tenu de la contrainte extrême : **moins d'une heure de données annotées disponibles**.

---

## Stack technique

| Composant | Outil |
|-----------|-------|
| Modèle ASR | Whisper (OpenAI) — architecture Transformer encodeur-décodeur |
| Framework | PyTorch + HuggingFace Transformers |
| Données | HuggingFace Datasets, FLEURS, Common Voice |
| Métriques | Jiwer (WER/CER), HuggingFace Evaluate |
| Environnement | Python 3.10, GPU NVIDIA Quadro RTX 6000 (cluster LIUM) |
| Planificateur | SLURM (cluster Le Mans Université) |

---

## Perspectives

- **Généralisation à d'autres langues peu dotées** — tester l'approche pivot sur le Baloutche ou le Talysh (même famille iranienne) pour valider que la stratégie est transférable au-delà du zazaki.
- **Augmentation acoustique** — modifier la vitesse, ajouter du bruit ou changer le pitch pour diversifier artificiellement le corpus d'entraînement.
- **Synthèse vocale (TTS)** — générer de nouvelles paires audio/texte à partir de textes zazaki existants pour contourner la pénurie de données audio.
- **LoRA (Low-Rank Adaptation)** — fine-tuner uniquement un sous-ensemble des paramètres de Whisper pour stabiliser l'apprentissage sur très petits corpus et réduire le risque d'overfitting.
- **Pseudo-labeling itératif** — répéter le cycle (inférence → filtrage → ré-entraînement) plusieurs fois pour continuer à améliorer le modèle de façon autonome.

---

<div align="center">

**Réalisé à Le Mans Université — M1 Intelligence Artificielle, 2025–2026**  
Module Ingénierie R&D — Semestre 1

*ASR · Whisper · Langues peu dotées · Transfer Learning · Pseudo-labeling*

</div>
