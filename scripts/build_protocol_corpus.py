"""Indexe les protocoles de rééducation structurés dans ChromaDB.

Crée la collection "aria_protocols" (distincte de "aria_pubmed").
Chaque exercice = 1 document avec les métadonnées :
    pathologie, phase, semaines, exercice, source

Sources : protocoles Alfredson, Fredericson, Crossley, McGill,
          CPG JOSPT (2012–2021).

Usage:
    uv run python scripts/build_protocol_corpus.py
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

from core.config import settings

COLLECTION_NAME = "aria_protocols"
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64
E5_PASSAGE_PREFIX = "passage: "

# ---------------------------------------------------------------------------
# Données des protocoles
# Chaque entrée : (slug_pathologie, phase, semaines, exercice, description, source)
# ---------------------------------------------------------------------------

_PROTOCOLS: list[tuple[str, int, str, str, str, str]] = [

    # ── LOMBALGIE ──────────────────────────────────────────────────────────
    # Phase 1 — Stabilisation motrice (McGill Big 3 + activation profonde)
    ("lombalgie", 1, "S1–S2", "McGill Curl-up",
     "Décubitus dorsal, un genou fléchi. Mains sous les lombaires pour maintenir la lordose. "
     "Élever légèrement tête et épaules sans creuser le dos. 3×10, 8 s de maintien. "
     "Objectif : activation des abdominaux superficiels sans flexion lombaire.",
     "McGill 2010 – Low Back Disorders"),

    ("lombalgie", 1, "S1–S2", "Side Bridge",
     "Décubitus latéral, appui sur le coude et les pieds empilés. "
     "Hanches décollées du sol, corps aligné. 3×10 s → progresser jusqu'à 30 s. "
     "Objectif : endurance du carré des lombes et obliques.",
     "McGill 2010 – Low Back Disorders"),

    ("lombalgie", 1, "S1–S2", "Bird-Dog",
     "Quadrupédie, dos plat (activation transverse de l'abdomen). "
     "Étendre bras droit + jambe gauche simultanément, maintenir 8 s. "
     "3×10 répétitions par côté. Éviter la rotation du bassin.",
     "McGill 2010 – Low Back Disorders"),

    ("lombalgie", 1, "S1–S2", "Activation transverse de l'abdomen",
     "Décubitus dorsal ou debout. Inspiration diaphragmatique, expiration lente "
     "avec rentrée du bas-ventre ('draw-in'). 3×10 cycles, 5 s maintien. "
     "Prépare les stabilisateurs profonds avant toute charge.",
     "Richardson et al. 1999 – Therapeutic Exercise for Spinal Segmental Stabilization"),

    ("lombalgie", 1, "S1–S2", "Rétroversion pelvienne contrôlée",
     "Décubitus dorsal, genoux fléchis. Appuyer les lombaires au sol (bascule postérieure), "
     "tenir 5 s, relâcher. 3×15. Permet la conscientisation de la position neutre lombaire.",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 1, "S1–S2", "Pont fessier bilatéral",
     "Décubitus dorsal, pieds à plat. Soulever les hanches jusqu'à alignement épaule-hanche-genou. "
     "3×15, 3 s maintien au sommet. Renforce les ischio-jambiers et les fessiers.",
     "CPG JOSPT 2021 – Low Back Pain"),

    # Phase 2 — Renforcement fonctionnel
    ("lombalgie", 2, "S3–S5", "Squat goblet",
     "Charge légère devant la poitrine (haltère ou kettlebell). Descente contrôlée "
     "jusqu'à 90° genoux, dos neutre. 3×12. Renforce chaîne postérieure en maintenant "
     "la stabilité lombaire.",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 2, "S3–S5", "Hip hinge avec bande élastique",
     "Debout, bande élastique au niveau des hanches fixée en avant. "
     "Inclinaison du buste en avant (charnière hanche), dos neutre, genoux légèrement fléchis. "
     "3×15. Pattern fondamental pour les activités quotidiennes et la course.",
     "McGill 2010 – Low Back Disorders"),

    ("lombalgie", 2, "S3–S5", "Romanian Deadlift au poids du corps",
     "Debout, hanches en arrière, tronc incliné, dos plat. "
     "Descente jusqu'à tension dans les ischio-jambiers, remontée en activant les fessiers. "
     "3×12. Progression vers le deadlift lesté (phase 3).",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 2, "S3–S5", "Step-up avec contrôle du tronc",
     "Marche montante sur step (20 cm), buste droit, regard devant. "
     "Éviter l'inclinaison latérale du tronc. 3×10 par jambe. "
     "Transfert vers la chaîne unipodal.",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 2, "S3–S5", "Planche sur Swiss ball",
     "Avant-bras sur le ballon, corps en appui sur les orteils. "
     "Maintenir la position 20–30 s sans creuser les lombaires. "
     "3×5 répétitions. Variante instable du gainage.",
     "McGill 2010 – Low Back Disorders"),

    ("lombalgie", 2, "S3–S5", "Pont fessier unilatéral",
     "Décubitus dorsal, une jambe levée (genou tendu). "
     "Soulever les hanches avec la jambe d'appui. 3×12 par côté. "
     "Renforce les fessiers unilatéralement, prépare la phase unipodal de la course.",
     "CPG JOSPT 2021 – Low Back Pain"),

    # Phase 3 — Charge progressive + retour course
    ("lombalgie", 3, "S6–S10", "Deadlift barre légère",
     "Barre au sol, prise en pronation, dos plat. Extension hanche-genou simultanée. "
     "3×8, charge à 50–60 % 1RM. Progrès hebdomadaire de 5 %. "
     "Critère d'entrée : douleur ≤ 2/10 lors du hip hinge phase 2.",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 3, "S6–S10", "Single-leg Romanian Deadlift",
     "Debout sur une jambe, buste incliné vers l'avant, jambe libre en arrière. "
     "Haltère dans la main controlatérale. 3×10 par côté. "
     "Proprioception + chaîne postérieure unipodal.",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 3, "S6–S10", "Retour course progressif avec métronome",
     "Objectif cadence 175 spm avec métronome. Séances alternées marche 2 min / course 3 min. "
     "Augmentation de 10 % du volume hebdomadaire. Surveiller douleur lombaire NRS après course.",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 3, "S6–S10", "Lateral band walk",
     "Bande élastique aux chevilles ou genoux. Pas latéraux en maintenant les genoux "
     "légèrement fléchis et le tronc droit. 3×15 pas par côté. "
     "Renforce les abducteurs de hanche, réduit l'oscillation latérale.",
     "CPG JOSPT 2021 – Low Back Pain"),

    ("lombalgie", 3, "S6–S10", "Sled push (traîneau ou charge au sol)",
     "Poussée horizontale 10–15 m avec dos neutre, activation fessiers et stabilisateurs. "
     "3 séries. Renforcement fonctionnel en chaîne fermée mimant la propulsion en course.",
     "McGill 2010 – Low Back Disorders"),

    ("lombalgie", 3, "S6–S10", "Test retour course : 30 min sans douleur",
     "Sortie continue 30 min à allure conversationnelle sur terrain plat. "
     "Critère de validation : NRS ≤ 2/10 pendant et dans les 24 h suivant la séance. "
     "Autorisation de reprise compétition si validé 2 séances consécutives.",
     "CPG JOSPT 2021 – Low Back Pain"),

    # ── SFP / TENDINITE ROTULIENNE ─────────────────────────────────────────
    # Phase 1 — Décharge + activation VMO
    ("sfp", 1, "S1–S3", "Isométrie quadriceps à 45°",
     "Leg press ou mur, genou à 45°. Contraction maximale maintenue 45 s × 5 répétitions. "
     "Effet analgésique immédiat (réduction douleur rotulienne NRS -2 points). "
     "Crossley et al. recommandent 4 séries/jour en phase aiguë.",
     "Crossley et al. 2016 – BJSM Patellofemoral Pain"),

    ("sfp", 1, "S1–S3", "Mini squat (0–30°)",
     "Descente limitée à 30° de flexion genou pour éviter la surcharge rotulienne. "
     "Accent sur activation du vaste médial oblique (VMO). 3×15 répétitions. "
     "Progresser vers 60° à la phase 2.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 1, "S1–S3", "Straight Leg Raise (SLR)",
     "Décubitus dorsal, jambe saine fléchie. Contracter le quadriceps de la jambe tendue "
     "et élever jusqu'à 45°. 3×15, 2 s maintien au sommet. "
     "Active le quadriceps sans contrainte rotulo-fémorale.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 1, "S1–S3", "Clamshell (activation abducteurs hanche)",
     "Décubitus latéral, genoux fléchis à 45°, pieds empilés. "
     "Ouvrir le genou supérieur sans tourner le bassin. 3×20. "
     "Renforce le grand fessier et le moyen fessier, réduit le valgus dynamique.",
     "Crossley et al. 2016 – BJSM Patellofemoral Pain"),

    ("sfp", 1, "S1–S3", "Tape rotulien (McConnell taping)",
     "Application du tape médial sur la rotule pour corriger son tracking. "
     "Réduit la douleur de 50 % en moyenne (McConnell 1986). "
     "Permet la réalisation des exercices en phase aiguë. Retirer si irritation cutanée.",
     "McConnell 1986 – Australian Journal of Physiotherapy"),

    ("sfp", 1, "S1–S3", "Vélo stationnaire (sans résistance)",
     "20–30 min à faible résistance, selle haute (genou < 70° flexion). "
     "Maintien aérobie sans surcharge rotulo-fémorale. "
     "Arrêt si douleur > 3/10 pendant l'exercice.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    # Phase 2 — Renforcement chaîne cinétique
    ("sfp", 2, "S4–S7", "Step-down excentrique",
     "Debout sur step (15–20 cm), descendre lentement sur une jambe jusqu'à "
     "talonnage léger sur le sol. 3×15 par jambe. Contrôle valgus du genou. "
     "Exercice clé Crossley pour le retour fonctionnel.",
     "Crossley et al. 2016 – BJSM Patellofemoral Pain"),

    ("sfp", 2, "S4–S7", "Leg press (0–60°)",
     "Presse à jambes, amplitude 0–60° pour limiter la contrainte rotulo-fémorale. "
     "3×12 avec charge modérée. Progression : augmentation amplitude jusqu'à 90° "
     "si pas de douleur.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 2, "S4–S7", "Terminal Knee Extension (TKE) avec bande",
     "Bande élastique derrière le genou. Extension terminale (0–30°) contre résistance. "
     "3×20 répétitions. Cible le VMO en fin d'extension.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 2, "S4–S7", "Hip Thrust",
     "Épaules sur banc, pieds à plat, poussée des hanches vers le haut. "
     "3×12, charge progressive (haltère ou barre). "
     "Renforce le grand fessier, réduit les contraintes rotulo-fémorales en course.",
     "Crossley et al. 2016 – BJSM Patellofemoral Pain"),

    ("sfp", 2, "S4–S7", "Single-leg squat partiel",
     "Squat unilatéral jusqu'à 45°, contrôle de l'alignement genou sur 2e orteil. "
     "3×10 par jambe. Critère : absence de douleur > 2/10. "
     "Progression vers profondeur totale en phase 3.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 2, "S4–S7", "Renforcement abducteurs hanche en résistance",
     "Side-lying abduction avec bande élastique à la cheville. "
     "3×15 par côté. Adjonction d'un poids pour progression. "
     "Vise à réduire l'adduction de hanche et le valgus dynamique en course.",
     "Nakagawa et al. 2012 – Journal of Orthopaedic & Sports Physical Therapy"),

    # Phase 3 — Retour course
    ("sfp", 3, "S8–S12", "Retour course progressive (cadence 175 spm)",
     "Métronome à 175 spm. Séances marche 1 min / course 2 min × 10 cycles. "
     "Augmentation cadence de 5–10 % réduit les contraintes rotulo-fémorales (Lenhart 2014). "
     "Critère d'entrée : step-down sans douleur > 2/10.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 3, "S8–S12", "Box step-up progressif",
     "Step-up sur boîte de 30 cm, progression 30→40→50 cm. "
     "3×10 par jambe, montée contrôlée et descente lente. "
     "Simule l'attaque au sol en course sur le plan fonctionnel.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 3, "S8–S12", "Lateral lunges",
     "Fente latérale avec contrôle du genou sur l'orteil. 3×12 par côté. "
     "Travail en plan frontal, renforce les stabilisateurs de hanche.",
     "Crossley et al. 2016 – BJSM Patellofemoral Pain"),

    ("sfp", 3, "S8–S12", "Short foot exercise (contrôle de la pronation)",
     "Debout, contracter l'arche plantaire sans fléchir les orteils. "
     "3×30 s, puis dynamique pendant la marche. "
     "Réduit la pronation excessive qui augmente le valgus et la contrainte rotulienne.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 3, "S8–S12", "Double vers single-leg landing",
     "Saut vertical bilatéral, atterrissage en douceur deux jambes puis une jambe. "
     "3×10 répétitions. Contrôle de l'angle du genou à l'atterrissage (< valgus). "
     "Test fonctionnel avant reprise compétition.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    ("sfp", 3, "S8–S12", "Sortie course 30 min — critère libération",
     "Course continue 30 min sur terrain plat, allure conversationnelle. "
     "NRS ≤ 2/10 pendant + absence de douleur résiduelle à J+1. "
     "Critère de libération compétition si validé 2 fois consécutives.",
     "CPG JOSPT 2019 – Patellofemoral Pain"),

    # ── SBIT ───────────────────────────────────────────────────────────────
    # Phase 1 — Décharge + assouplissement
    ("sbit", 1, "S1–S2", "Foam rolling bandelette ilio-tibiale",
     "Rouleau en mousse sur la face latérale de cuisse, 60–90 s par zone sensible. "
     "Éviter de rouler directement sur la tête du péroné. "
     "Réduit la tension mécanique de la BIT en phase aiguë.",
     "Fredericson & Wolf 2005 – Physical Medicine and Rehabilitation"),

    ("sbit", 1, "S1–S2", "IT band stretch debout (cross-leg)",
     "Jambe à traiter derrière l'autre, pencher le buste vers la jambe saine "
     "en appuyant la hanche contre le mur. Maintenir 30 s × 3. "
     "Ressenti d'étirement face latérale de cuisse.",
     "Fredericson & Wolf 2005 – Physical Medicine and Rehabilitation"),

    ("sbit", 1, "S1–S2", "Ober's stretch (décubitus latéral)",
     "Décubitus latéral côté sain, jambe à traiter en extension et adduction passive "
     "par l'évaluateur ou une sangle. 30 s × 3. "
     "Étire spécifiquement la BIT et le tenseur du fascia lata.",
     "Noble 1980 – Physician and Sportsmedicine"),

    ("sbit", 1, "S1–S2", "Hip adductor stretch en fente latérale",
     "Fente latérale lente, genou de la jambe tendue en alignement. "
     "30 s × 3 par côté. Relâche les tensions latérales par réciprocité.",
     "CPG JOSPT 2012 – ITBS"),

    ("sbit", 1, "S1–S2", "Clamshell (activation fessier moyen)",
     "Décubitus latéral, genoux fléchis. Ouvrir le genou supérieur. "
     "3×20, progression avec bande élastique. "
     "Prépare le renforcement actif de la phase 2.",
     "Fredericson & Wolf 2005 – Physical Medicine and Rehabilitation"),

    ("sbit", 1, "S1–S2", "Vélo stationnaire ou aqua-jogging",
     "Alternative à la course pendant la phase aiguë. "
     "30 min vélo (résistance légère) ou 20 min aqua-jogging avec gilet. "
     "Maintient la capacité cardiovasculaire sans contrainte latérale.",
     "CPG JOSPT 2012 – ITBS"),

    # Phase 2 — Renforcement hanche (Fredericson protocol)
    ("sbit", 2, "S3–S5", "Side-lying hip abduction (Fredericson étape 1)",
     "Décubitus latéral, abduction de la jambe supérieure à 30°. "
     "3×15 par côté. Renforce le moyen fessier. "
     "Fredericson 2000 : réduction SBIT à 6 semaines avec ce protocole.",
     "Fredericson et al. 2000 – Clinical Journal of Sport Medicine"),

    ("sbit", 2, "S3–S5", "Standing hip abduction avec bande (Fredericson étape 2)",
     "Debout, bande élastique à la cheville, abduction de la jambe traitée. "
     "3×15 par côté. Progression en charge par rapport à la phase allongée.",
     "Fredericson et al. 2000 – Clinical Journal of Sport Medicine"),

    ("sbit", 2, "S3–S5", "Single-leg squat (Fredericson étape 3)",
     "Squat sur une jambe jusqu'à 45°. Contrôle du pelvic drop et du valgus genou. "
     "3×10 par côté. Critère d'entrée en phase 3.",
     "Fredericson et al. 2000 – Clinical Journal of Sport Medicine"),

    ("sbit", 2, "S3–S5", "Wall squat avec ballon (contrôle valgus)",
     "Ballon entre le genou et le mur, descente lente. Isométrique 30 s × 5. "
     "Active l'adducteur tout en éduquant l'alignement du genou.",
     "CPG JOSPT 2012 – ITBS"),

    ("sbit", 2, "S3–S5", "Lateral band walk",
     "Bande élastique aux genoux ou chevilles. 15 pas latéraux × 3 séries. "
     "Renforce les abducteurs de hanche en pattern fonctionnel.",
     "Fredericson & Wolf 2005 – Physical Medicine and Rehabilitation"),

    ("sbit", 2, "S3–S5", "Nordic hamstring curl",
     "Agenouillé, pieds fixés, descendre le buste vers le sol excentrically. "
     "3×6–8. Renforce les ischio-jambiers pour le contrôle du genou en phase d'appui.",
     "CPG JOSPT 2012 – ITBS"),

    # Phase 3 — Retour course
    ("sbit", 3, "S6–S10", "Running retraining — augmentation cadence 5–10 %",
     "Métronome : augmenter la cadence de 5 à 10 % par rapport à la cadence spontanée. "
     "Réduit l'adduction de hanche et la contrainte sur la BIT (Noehren 2011). "
     "Séances de 20 min, 3×/semaine.",
     "Noehren et al. 2011 – Clinical Biomechanics"),

    ("sbit", 3, "S6–S10", "Retour course progressif (2 → 5 → 10 km)",
     "Semaine 1 : 2 km plat. Semaine 2 : 5 km. Semaine 3 : 10 km. "
     "Éviter les virages serrés répétés et les terrains bombés. "
     "Critère : douleur ≤ 2/10 NRS pendant et après.",
     "Fredericson & Wolf 2005 – Physical Medicine and Rehabilitation"),

    ("sbit", 3, "S6–S10", "Réintroduction descentes (graduelle)",
     "Commencer par des pentes < 5 % en courant. "
     "Augmenter progressivement sur 2 semaines. "
     "La descente augmente la tension BIT : à réintroduire en dernier.",
     "CPG JOSPT 2012 – ITBS"),

    ("sbit", 3, "S6–S10", "Single-leg bridge progressif",
     "Pont fessier sur une jambe, progression avec charge sur les hanches. "
     "3×15 par côté. Maintien du renforcement fessier pendant le retour course.",
     "Fredericson et al. 2000 – Clinical Journal of Sport Medicine"),

    ("sbit", 3, "S6–S10", "Lateral trunk lean retraining",
     "Travail devant miroir ou vidéo pour éliminer l'inclinaison latérale du tronc "
     "côté portant. Réduit le pelvic drop et la tension BIT. "
     "10 min par séance de course.",
     "Noehren et al. 2011 – Clinical Biomechanics"),

    ("sbit", 3, "S6–S10", "Test sortie 10 km — critère libération",
     "Course 10 km terrain plat, allure modérée. NRS ≤ 2/10 pendant et à J+1. "
     "Validé si 2 sorties consécutives sans douleur.",
     "Fredericson & Wolf 2005 – Physical Medicine and Rehabilitation"),

    # ── PÉRIOSTITE TIBIALE ─────────────────────────────────────────────────
    # Phase 1 — Décharge + renforcement intrinsèque
    ("periostite_tibiale", 1, "S1–S3", "Calf raises bilatéraux",
     "Debout sur le bord d'un step, amplitude complète (talon bas → pointe haute). "
     "3×20 répétitions lentes (2 s montée, 3 s descente). "
     "Active le triceps sural et le soléaire, réduit les contraintes tibiales.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 1, "S1–S3", "Tibialis anterior strengthening (Toe raises)",
     "Debout, dos au mur, élever la pointe des pieds. 3×20. "
     "Renforce le tibial antérieur, antagoniste du triceps sural, "
     "équilibre la musculature de jambe.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 1, "S1–S3", "Short foot exercise",
     "Assis ou debout, raccourcir l'arche plantaire sans fléchir les orteils. "
     "3×30 s par pied. Renforce les muscles intrinsèques du pied, "
     "améliore le contrôle de la pronation.",
     "Mulligan & Cook 2013 – Manual Therapy"),

    ("periostite_tibiale", 1, "S1–S3", "Marble pickup (flexeurs orteils)",
     "Assis, ramasser des billes avec les orteils et les déposer dans un bol. "
     "3×20 répétitions. Renforce les fléchisseurs courts des orteils, "
     "stabilise l'arche médiale.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 1, "S1–S3", "Pool walking / vélo (maintien aérobie)",
     "Marche aquatique 20–30 min ou vélo sans résistance 30 min, 3×/semaine. "
     "Maintient la capacité cardiovasculaire sans impact tibial. "
     "Poursuivre jusqu'à absence de douleur à la palpation tibiale.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 1, "S1–S3", "Étirements triceps sural (Gastrocnemius + Soléaire)",
     "Gastrocnemius : genou tendu, pied contre le mur, 3×30 s. "
     "Soléaire : genou fléchi à 30°, même position, 3×30 s. "
     "Réduit la tension mécanique sur le périoste tibial.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    # Phase 2 — Correction technique + renforcement
    ("periostite_tibiale", 2, "S4–S7", "Cadence drill avec métronome",
     "Augmenter la cadence de 5–10 % (cible 175–180 spm). "
     "Réduit l'amplitude du pas et les forces d'impact au sol de 20 % (Heiderscheit 2011). "
     "Séances de 20 min, 3×/semaine.",
     "Heiderscheit et al. 2011 – Medicine & Science in Sports & Exercise"),

    ("periostite_tibiale", 2, "S4–S7", "Single-leg calf raise excentrique",
     "Sur step, une jambe, descente lente du talon (3 s). "
     "3×15 répétitions. Renforce le soléaire et le gastrocnemius en excentrique, "
     "augmente la tolérance à la charge tibiale.",
     "Moen et al. 2012 – British Journal of Sports Medicine"),

    ("periostite_tibiale", 2, "S4–S7", "Renforcement abducteurs hanche (Clamshell avancé)",
     "Clamshell avec bande élastique, 3×20. "
     "Puis standing abduction avec bande. "
     "Réduit la pronation excessive en contrôlant le valgus dynamique.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 2, "S4–S7", "Romanian Single-leg Deadlift",
     "Balance sur une jambe, descente du buste, haltère dans la main controlatérale. "
     "3×12 par côté. Proprioception + chaîne postérieure.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 2, "S4–S7", "Jump rope progressif (surface souple)",
     "Corde à sauter sur tapis mousse ou herbe. 2×2 min, progression sur 3 semaines. "
     "Réintroduction de l'impact contrôlé en avant-pied.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 2, "S4–S7", "Forefoot strike retraining",
     "Rééducation attaque avant-pied devant miroir et vidéo. "
     "Réduit la sollicitation tibiale médiale de 30 % (Daoud 2012). "
     "Introduire progressivement sur 4 semaines.",
     "Daoud et al. 2012 – Medicine & Science in Sports & Exercise"),

    # Phase 3 — Retour course
    ("periostite_tibiale", 3, "S8–S12", "Programme retour course walk-jog",
     "Semaine 1 : 1 min marche / 1 min course × 15. "
     "Semaine 2 : 1 min marche / 2 min course × 12. "
     "Semaine 3 : 1 min marche / 5 min course × 6. "
     "Critère : douleur tibiale ≤ 2/10 NRS.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 3, "S8–S12", "Maintien cadence 175–180 spm",
     "Vérification de la cadence en course avec montre GPS ou métronome. "
     "Cible : attaque médio-pied, pas de talon excessif. "
     "Bilan vidéo à S8 et S12.",
     "Heiderscheit et al. 2011 – Medicine & Science in Sports & Exercise"),

    ("periostite_tibiale", 3, "S8–S12", "Surface variée progressive",
     "Débuter sur herbe/piste synthétique. Intégrer asphalte après 2 semaines de tolérance. "
     "Éviter les surfaces dures exclusives en phase de reprise.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 3, "S8–S12", "Volume +10 % par semaine maximum",
     "Règle des 10 % : ne jamais dépasser +10 % de volume hebdomadaire. "
     "Tenir un journal de course (km, intensité, douleur). "
     "Réduire le volume si NRS > 3 après sortie.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    ("periostite_tibiale", 3, "S8–S12", "Single-leg hop test (critère retour sport)",
     "3 sauts unipodaux pour distance. Critère : symétrie ≥ 90 % (jambe lésée / saine). "
     "Si non atteint : maintenir renforcement 2 semaines supplémentaires.",
     "Moen et al. 2012 – British Journal of Sports Medicine"),

    ("periostite_tibiale", 3, "S8–S12", "Plyométrie légère (skip A et B)",
     "Skip A : montée genou alternée avec élan. Skip B : extension de la jambe. "
     "2×30 m, 3×/semaine. Prépare le système nerveux à la charge explosive.",
     "CPG JOSPT 2019 – Medial Tibial Stress Syndrome"),

    # ── TENDINITE ACHILLE ──────────────────────────────────────────────────
    # Phase 1 — Isométrique + charge initiale (Alfredson)
    ("tendinite_achille", 1, "S1–S4", "Heel drop excentrique genou tendu (Alfredson)",
     "Debout sur le bord d'un step. Monter sur deux jambes, descendre lentement "
     "sur la jambe concernée en laissant le talon passer sous le niveau du step. "
     "3×15, 2×/jour, 7 jours/7. Douleur acceptée ≤ 5/10. "
     "Protocole Alfredson 1998 : 90 % de succès à 12 semaines.",
     "Alfredson et al. 1998 – American Journal of Sports Medicine"),

    ("tendinite_achille", 1, "S1–S4", "Heel drop excentrique genou fléchi (Alfredson soléaire)",
     "Même exercice que précédent mais genou fléchi à 30°. "
     "3×15, 2×/jour. Cible spécifiquement le soléaire. "
     "Les deux variantes (genou tendu + fléchi) sont nécessaires.",
     "Alfredson et al. 1998 – American Journal of Sports Medicine"),

    ("tendinite_achille", 1, "S1–S4", "Isométrie triceps sural (45 s × 5)",
     "Leg press ou step, contraction isométrique à 25° de plantarflexion. "
     "45 s × 5 répétitions, 2 min repos entre séries. "
     "Effet analgésique tendino-spécifique (Rio et al. 2015).",
     "Rio et al. 2015 – British Journal of Sports Medicine"),

    ("tendinite_achille", 1, "S1–S4", "Towel stretch (étirement passif triceps sural)",
     "Assis, serviette autour du pied, genou tendu. "
     "Traction vers soi, maintien 30 s × 3. "
     "Maintient la souplesse sans exercer de traction sur le tendon.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 1, "S1–S4", "Calf raise bilatéral concentrique (warm-up)",
     "Montée sur la pointe des pieds, bilatéral, 3×20. "
     "Sert de warm-up avant les excentriques. "
     "Pas de descente excentrique à ce stade (réservé à la phase 2).",
     "Alfredson et al. 1998 – American Journal of Sports Medicine"),

    ("tendinite_achille", 1, "S1–S4", "Vélo stationnaire (maintien aérobie)",
     "30–45 min de vélo sans résistance excessive. "
     "Préserver la capacité aérobie pendant la phase de décharge. "
     "Arrêt si douleur > 3/10 à l'avant du pied.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    # Phase 2 — Charge progressive (HSR)
    ("tendinite_achille", 2, "S5–S9", "Heavy Slow Resistance — leg press calf unilatéral",
     "Presse à jambes, une jambe. Charge permettant 6 répétitions (6RM). "
     "3×6, rythme 3 s concentrique / 4 s excentrique. "
     "HSR = équivalent Alfredson excentrique (Beyer 2015 : mêmes résultats).",
     "Beyer et al. 2015 – American Journal of Sports Medicine"),

    ("tendinite_achille", 2, "S5–S9", "Seated calf raise lestée",
     "Machine ou sac lesté sur les genoux, assis. "
     "3×12, rythme lent. Cible isolément le soléaire (tendon distal). "
     "Progresser en charge chaque semaine (+2.5 kg).",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 2, "S5–S9", "Excentrique-concentrique heel drop (transition)",
     "Descente excentrique + remontée concentrique sur la même jambe. "
     "3×12. Transition progressive du protocole Alfredson pur vers la charge complète.",
     "Alfredson et al. 1998 – American Journal of Sports Medicine"),

    ("tendinite_achille", 2, "S5–S9", "Single-leg balance (proprioception)",
     "Appui unipodal yeux ouverts 30 s → yeux fermés 30 s → sur surface instable. "
     "3 séries par côté. Proprioception essentielle après tendinopathie.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 2, "S5–S9", "Renforcement hanche (pont fessier + clamshell)",
     "Pont fessier 3×15 + clamshell avec bande 3×20. "
     "Réduit la surcharge sur le triceps sural en améliorant la propulsion de hanche.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 2, "S5–S9", "Walk-jog intervals (si NRS ≤ 2/10 au matin)",
     "Critère d'entrée : douleur matinale NRS ≤ 2/10. "
     "1 min marche / 1 min trot léger × 10. Progresser sur 3 semaines. "
     "Ne pas dépasser 3 séances/semaine en phase mixte.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    # Phase 3 — Retour course + plyométrie
    ("tendinite_achille", 3, "S10–S16", "Running retraining cadence 175 spm",
     "Métronome à 175 spm. Attaque médio-pied progressive. "
     "Réduit la dorsiflexion brusque et la sollicitation du tendon d'Achille. "
     "Surveiller la douleur en fin de séance et le lendemain matin.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 3, "S10–S16", "Single-leg calf raise explosive",
     "Montée rapide sur la pointe du pied, une jambe. 3×10. "
     "Développe la puissance élastique du tendon. "
     "Critère d'entrée : 25 répétitions unilatérales sans douleur.",
     "Beyer et al. 2015 – American Journal of Sports Medicine"),

    ("tendinite_achille", 3, "S10–S16", "Pogo jumps bilatéraux",
     "Sauts sur place, genou quasi-tendu, contact sol le plus bref possible. "
     "3×20. Sollicite le cycle étirement-détente du tendon d'Achille. "
     "Introduire progressivement en 2 semaines.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 3, "S10–S16", "Retour course (protocole Couch-to-5K modifié)",
     "Semaine 10 : 2×10 min trot. Semaine 12 : 2×20 min. Semaine 14 : 30 min continu. "
     "Progression limitée à +10 %/semaine. NRS ≤ 2/10 pendant et à J+1.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 3, "S10–S16", "Sprint drills A-skip et B-skip",
     "Skip A : montée genou + frappe du pied au sol. Skip B : extension complète. "
     "2×20 m, 3×/semaine. Prépare les efforts explosifs de la compétition.",
     "CPG JOSPT 2018 – Achilles Tendinopathy"),

    ("tendinite_achille", 3, "S10–S16", "Test single-leg hop for distance (critère libération)",
     "3 sauts unipodaux pour distance maximale. "
     "Critère : symétrie ≥ 90 % entre les deux jambes. "
     "Validé si douleur NRS ≤ 2/10 à J+1.",
     "Beyer et al. 2015 – American Journal of Sports Medicine"),

    # ── FASCIITE PLANTAIRE ─────────────────────────────────────────────────
    # Phase 1 — Décharge + étirements ciblés
    ("fasciite_plantaire", 1, "S1–S3", "Plantar fascia stretch au réveil",
     "Assis au bord du lit, avant de poser le pied. Saisir les orteils, "
     "extension dorsale maximale, maintenir 10 s × 10. "
     "À réaliser avant chaque lever. DiGiovanni 2003 : 52 % réduction douleur à 8 semaines.",
     "DiGiovanni et al. 2003 – Journal of Bone and Joint Surgery"),

    ("fasciite_plantaire", 1, "S1–S3", "Calf stretch Gastrocnemius (genou tendu)",
     "Pied contre le mur, jambe tendue, talon au sol. 3×30 s. "
     "Réduit la tension sur le fascia plantaire via la restriction en dorsiflexion.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 1, "S1–S3", "Calf stretch Soléaire (genou fléchi)",
     "Pied contre le mur, genou fléchi à 20–30°, talon au sol. 3×30 s. "
     "Cible spécifiquement le soléaire, souvent plus restrictif que le gastrocnemius.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 1, "S1–S3", "Short foot exercise (renforcement intrinsèque)",
     "Raccourcir l'arche du pied sans fléchir les orteils. "
     "3×30 s statique → progresser vers dynamique en marche. "
     "Réduit la pronation excessive qui surcharge le fascia.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 1, "S1–S3", "Frozen bottle rolling (automassage plantaire)",
     "Rouler une bouteille d'eau gelée sous la voûte plantaire 5 min. "
     "Effet analgésique par cryothérapie locale + mobilisation du fascia.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 1, "S1–S3", "Night splint (si douleur matinale > 5/10)",
     "Attelle de dorsiflexion nocturne (5° dorsiflexion). "
     "Maintient l'étirement passif du fascia pendant le sommeil. "
     "Indiqué si douleur première marche matinale sévère.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    # Phase 2 — Renforcement intrinsèque + fascio-tendineux
    ("fasciite_plantaire", 2, "S4–S8", "High-load single-leg heel raise sur step (Rathleff)",
     "Debout sur step, une jambe, serviette roulée sous les orteils en extension. "
     "Montée lente, descente lente 3 s. 3×12, progression hebdomadaire en sac lesté. "
     "Rathleff 2015 : supérieur aux étirements seuls à 3 mois.",
     "Rathleff et al. 2015 – Scandinavian Journal of Medicine & Science in Sports"),

    ("fasciite_plantaire", 2, "S4–S8", "Toe curls avec serviette",
     "Serviette à plat, plisser avec les orteils. 3×20 répétitions. "
     "Renforce les fléchisseurs courts des orteils, soutient l'arche médiale.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 2, "S4–S8", "Single-leg calf raise progressif",
     "Bilatéral → unilatéral, amplitude complète. 3×20 → 3×15 lesté. "
     "Renforce le triceps sural qui partage la charge avec le fascia plantaire.",
     "Rathleff et al. 2015 – Scandinavian Journal of Medicine & Science in Sports"),

    ("fasciite_plantaire", 2, "S4–S8", "Renforcement abducteurs hanche",
     "Clamshell + standing abduction avec bande. 3×20 par côté. "
     "Contrôle la pronation en améliorant la stabilité de hanche.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 2, "S4–S8", "Short foot exercise dynamique",
     "Maintien de l'arche pendant la marche et la montée d'escalier. "
     "Progression : même maintien lors de single-leg squat partiel. "
     "Transition entre exercice statique et fonctionnel.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 2, "S4–S8", "Taping anti-pronation (Low-Dye)",
     "Taping plantaire Low-Dye ou calcanéen. Réduit la tension sur le fascia de 30 %. "
     "Utiliser en complément des exercices, pas en substitution.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    # Phase 3 — Retour course
    ("fasciite_plantaire", 3, "S9–S16", "Running retraining cadence 175–180 spm",
     "Augmenter cadence de 5–10 %. Réduit la dorsiflexion de cheville et les forces "
     "au sol, diminuant la sollicitation du fascia (Willson 2014). "
     "Commencer à 15 min de course continue.",
     "Willson et al. 2014 – Journal of Orthopaedic & Sports Physical Therapy"),

    ("fasciite_plantaire", 3, "S9–S16", "Retour course surface souple progressif",
     "Herbe ou piste synthétique en priorité. Asphalte réintégré après 2 semaines. "
     "Volume : +10 % par semaine. Journal de douleur quotidien (NRS matin).",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 3, "S9–S16", "Single-leg squat avec contrôle pied",
     "Squat sur une jambe, maintien de l'arche plantaire active (short foot). "
     "3×12 par jambe. Test fonctionnel et exercice de transfert.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),

    ("fasciite_plantaire", 3, "S9–S16", "Réintroduction avant-pied progressif (si approprié)",
     "Si attaque talon excessive, rééduquer vers médio-pied sur 4 semaines. "
     "Ne pas forcer l'avant-pied complet (augmente la tension fascia distalement). "
     "Évaluer au cas par cas selon morphologie.",
     "Willson et al. 2014 – Journal of Orthopaedic & Sports Physical Therapy"),

    ("fasciite_plantaire", 3, "S9–S16", "Test fonctionnel : montée sur la pointe unilatérale",
     "25 répétitions single-leg heel raise sans douleur > 2/10. "
     "Critère de retour course compétition (Rathleff 2015).",
     "Rathleff et al. 2015 – Scandinavian Journal of Medicine & Science in Sports"),

    ("fasciite_plantaire", 3, "S9–S16", "Sortie 30 min sans douleur — critère libération",
     "Course continue 30 min, allure conversationnelle, terrain plat. "
     "NRS ≤ 2/10 pendant et NRS matin ≤ 2/10 à J+1 et J+2. "
     "Validé si 2 sorties consécutives.",
     "CPG JOSPT 2014 – Heel Pain / Plantar Fasciitis"),
]


def _doc_text(exercice: str, description: str, semaines: str, pathologie: str) -> str:
    """Construit le texte du document pour l'embedding."""
    return f"Pathologie : {pathologie}. Période : {semaines}. Exercice : {exercice}. {description}"


def _chroma_id(slug: str, phase: int, idx: int) -> str:
    return f"{slug}_p{phase}_{idx:03d}"


def batch(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def build_protocol_corpus() -> None:
    print(f"Chargement du modèle {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)

    chroma_path = Path(settings.CHROMADB_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    print(f"Documents déjà indexés dans '{COLLECTION_NAME}' : {len(existing_ids)}\n")

    # Compter les exercices par slug pour construire les IDs séquentiels
    slug_phase_counter: dict[tuple[str, int], int] = defaultdict(int)

    records = []
    for slug, phase, semaines, exercice, description, source in _PROTOCOLS:
        slug_phase_counter[(slug, phase)] += 1
        idx = slug_phase_counter[(slug, phase)]
        doc_id = _chroma_id(slug, phase, idx)
        text = _doc_text(exercice, description, semaines, slug)
        records.append({
            "id": doc_id,
            "text": text,
            "metadata": {
                "pathologie": slug,
                "phase": phase,
                "semaines": semaines,
                "exercice": exercice,
                "source": source,
            },
        })

    new_records = [r for r in records if r["id"] not in existing_ids]
    print(f"Total exercices dans le corpus : {len(records)}")
    print(f"Nouveaux documents à indexer   : {len(new_records)}")

    if not new_records:
        print("Collection déjà à jour — aucun upsert nécessaire.")
    else:
        for i, chunk in enumerate(batch(new_records, BATCH_SIZE)):
            texts = [E5_PASSAGE_PREFIX + r["text"] for r in chunk]
            embeddings = model.encode(texts, normalize_embeddings=True).tolist()
            collection.upsert(
                ids=[r["id"] for r in chunk],
                documents=[r["text"] for r in chunk],
                embeddings=embeddings,
                metadatas=[r["metadata"] for r in chunk],
            )
            done = min((i + 1) * BATCH_SIZE, len(new_records))
            print(f"  {done}/{len(new_records)} indexés…")

    # Rapport final par pathologie
    print("\n--- Documents indexés par pathologie ---")
    raw = collection.get(include=["metadatas"])
    counts: dict[str, int] = defaultdict(int)
    for meta in raw["metadatas"] or []:
        counts[str(meta.get("pathologie", ""))] += 1
    for slug, count in sorted(counts.items()):
        print(f"  {slug:<40} {count:>4} exercices")
    print(f"  {'TOTAL':<40} {sum(counts.values()):>4}")


if __name__ == "__main__":
    build_protocol_corpus()
