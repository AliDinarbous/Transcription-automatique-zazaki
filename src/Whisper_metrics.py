"""
Importation necessaire:
WhisperProcessor: - Pour convertir l'audio en en spectrogrammes log-Mel.
                  - Pour transcire les tokens generes par le decodeur en texte.
WhisperForConditionalGeneration: - Pour telecharger le model whisper et l'instancier
load_dataset: - Pour charger un dataset, elle retourne un dict de deux cles, train contient les colonnes et num_arrows le nombre de lignes.
Audio: - Pour transformer un fichier audio en un tableau numpy
"""
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio

import evaluate

import os
import torch


### Remarque:
"""
- Les datasets commonvoice utilise dans les scripts fourni par huggingface ne sont plus supporter par load_dataset  
  SOLUTION: On charge les donnees sur le Disk et on les recupere avec load_dataset et on fait un petit traitement pour utiliser
  le script sans le modifier.  
- Pour l'inference en anglais, Le script utilise un dataset qui ne pose pas ce probleme.  
"""



def load_and_process_data (DATASETS_PATH, FOLDER, FILE_NAME):
    # Charger le dataset
    dataset = load_dataset('csv', data_files=os.path.join("..", "datasets", FOLDER, f"{FILE_NAME}.tsv"), sep="\t")
    PATH_TO_AUDIO = DATASETS_PATH + FOLDER + "clips/"
    # Creer la colonne audio qui contient le chemin complet + le nom de fichier qui reside dans la colonne path
    dataset["train"] = dataset["train"].map( lambda x: {"audio": PATH_TO_AUDIO + x["path"]})
    """
    cast_columns: change le type d'une colonne
    Audio: permet de transformer un fichier audio en un tableau numpy avece un echantillonage de 16000 khz
    donc la colonne audio contiendra le tableau numpy qui sera apres transformer en spectogramme mel-log
    """
    dataset["train"] = dataset["train"].cast_column("audio", Audio(sampling_rate=16_000))
    return dataset



def whisper_inference (DATASETS_PATH, FOLDER, FILE_NAME, TASK="transcribe", language="multilingue", nb=10):
    
    # get the WordErrorRate module and the CharacterErrorRate module
    wer_metric = evaluate.load("wer", module_type="metric")
    cer_metric = evaluate.load("cer", module_type="metric")

    
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
   
    if language == "multilingue":
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task=TASK)
    else:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=TASK)

    # Recuperer le dataset
    ds = load_and_process_data(DATASETS_PATH, FOLDER, FILE_NAME)

    # Pour stocker les resultats
    results = []

    # Pour transcire tous les audio
    iterator_ds = iter(ds["train"])

    for la_data in iterator_ds:
        # recuperer la colonne audio qui contient le tableau numpy qui represente l'audio
        input_speech = la_data["audio"]

        # transformer le tableau numpy en un spectogramme log-mel
        input_features = processor( input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features
        # generer les tokens
        predicted_ids = model.generate(input_features)
        # decoder les tokens (transcire)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        # stocker les resultats
        results.append({
            "path": la_data["path"], # chemin complet
            "sentence": la_data["sentence"],  # transcription reelle (etiquette)
            "predicted": transcription,  # transcription Whisper    
            "wer": wer_metric.compute(references=[la_data["sentence"]], predictions=[transcription]),   # calcule le      Word Error Rate pour chaque data
            "cer": cer_metric.compute(references=[la_data["sentence"]], predictions=[transcription])    # calcule le Character Error Rate pour chaque data
        })

        wer_metric.add(references=la_data["sentence"], predictions=transcription)    # calcule le      Word Error Rate pour chaque data
        cer_metric.add(references=la_data["sentence"], predictions=transcription)    # calcule le Character Error Rate pour chaque data

    return results, wer_metric, cer_metric



def whisper_inference_pour_quelques_lignes (DATASETS_PATH, FOLDER, FILE_NAME, TASK="transcribe", LANGUAGE="multilingue", nb=10):

    # get the WordErrorRate module and the CharacterErrorRate module
    wer_metric = evaluate.load("wer", module_type="metric")
    cer_metric = evaluate.load("cer", module_type="metric")

    
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    if LANGUAGE == "multilingue":
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(task=TASK)
    else:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    
    # Recuperer le dataset
    ds = load_and_process_data(DATASETS_PATH, FOLDER, FILE_NAME)

    data = ds["train"]
    # Pour stocker les resultats
    results = []

    for ii in range(nb):
        # recuperer la colonne audio qui contient le tableau numpy qui represente l'audio
        input_speech = data[ii]["audio"]

        # transformer le tableau numpy en un spectogramme log-mel
        input_features = processor( input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features
        # generer les tokens
        predicted_ids = model.generate(input_features)
        # decoder les tokens (transcire)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # stocker les resultats
        results.append({
            "path": data[ii]["path"], # chemin complet
            "sentence": data[ii]["sentence"],  # transcription reelle (etiquette)
            "predicted": transcription,  # transcription Whisper    
            "wer": wer_metric.compute(references=[data[ii]["sentence"]], predictions=[transcription]),   # calcule le      Word Error Rate pour chaque data
            "cer": cer_metric.compute(references=[data[ii]["sentence"]], predictions=[transcription])    # calcule le Character Error Rate pour chaque data
        })

        wer_metric.add(references=data[ii]["sentence"], predictions=transcription)    # ajoute les références et prédictions pour chaque audio
        cer_metric.add(references=data[ii]["sentence"], predictions=transcription)    # ajoute les références et prédictions pour chaque audio
        

    return results, wer_metric, cer_metric


    
# boucle sur chaque phrase et transcription inférés + leurs WER et CER
def afficher_resultats_individuels(results):
    # affichage
    for r in results:
        #print(f"\n {r['path']}")
        print(f"Whisper: {r['predicted']}")
        print(f"label: {r['sentence']}")
        print(f"wer: {r['wer']}")
        print(f"cer: {r['cer']}")



# WER et CER calculées pendant l'inférence
def afficher_WER_CER_pour_chaque_donnees(results, display_path=True, display_predicted=True, display_reference=True, display_WER=True, display_CER=True):
    # affichage
    for r in results:
        if display_path:
            print(f"Path: {r['path']}")
        if display_predicted:
            print(f"Whisper: {r['predicted']}")
        if display_reference:
            print(f"Référence: {r['sentence']}")
        if display_WER:
            print(f"WER: {r['wer']}")
        if display_CER:
            print(f"CER: {r['cer']}")
        print("") # newline



# moyenne des WER et CER qui ont été calculées pendant l'inférence
# multiplié par 100 pour être un pourcentage?
def afficher_moyenne_globale_WER_CER(results):
    wer_sum = cer_sum = 0
    n = len(results)
    
    for r in results:
        wer_sum += r['wer']
        cer_sum += r['cer']

    # affichage
    print(f"Moyenne WordErrorRate: {(wer_sum/n)*100}")
    print(f"Moyenne CharErrorRate: {(cer_sum/n)*100}")



# "post inférence" parce que les métriques sont calculées après l'inférence, et non pendant
def afficher_calcul_WER_CER_post_inference(wer_metric, cer_metric):
    print(f"Calcul de WER à la fin: {wer_metric.compute()}")
    print(f"Calcul de CER à la fin: {cer_metric.compute()}")



    
# exemple d'exécution
def inference_example():


    # obtenir le chemin absolu vers les datasets
    os.echo("$USER")
    student_id = getpass.getuser()
    DATASETS_PATH = f"/info/raid-etu/m1/{student_id[0]}/projet-m1-asr/datasets/"
    if verbose == True:
        print(DATASETS_PATH)



    LANGUAGE = "french"
    TASK = "transcribe"
    FOLDER = "zazaki/"
    FILE_NAME = "test"
    """
    LANGUAGE = "en"
    TASK = "transcribe"
    FOLDER = "zazaki/"
    FILE_NAME = "test"
    """
    """
    LANGUAGE = "ar"
    TASK = "transcribe"
    FOLDER = "zazaki/"
    FILE_NAME = "test"
    """

    print("transcrire tous les audio : ")
    results, wer_metric, cer_metric = whisper_inference (DATASETS_PATH, LANGUAGE, TASK, FOLDER, FILE_NAME)

    afficher_WER_CER_pour_chaque_donnees(results, 
                                         display_path=False, 
                                         display_predicted=True, 
                                         display_reference=True, 
                                         display_WER=True, 
                                         display_CER=True)

    afficher_moyenne_globale_WER_CER(results)

    afficher_calcul_WER_CER_post_inference(wer_metric, cer_metric)



def inference_example_multilingue():


    # obtenir le chemin absolu vers les datasets
    os.echo("$USER")
    student_id = getpass.getuser()
    DATASETS_PATH = f"/info/raid-etu/m1/{student_id[0]}/projet-m1-asr/datasets/"
    if verbose == True:
        print(DATASETS_PATH)



    TASK = "transcribe"
    FOLDER = "zazaki/"
    FILE_NAME = "test"
    

    print("transcrire tous les audio : ")
    results, wer_metric, cer_metric = whisper_inference (DATASETS_PATH, TASK, FOLDER, FILE_NAME)

    afficher_WER_CER_pour_chaque_donnees(results, 
                                         display_path=False, 
                                         display_predicted=True, 
                                         display_reference=True, 
                                         display_WER=True, 
                                         display_CER=True)

    afficher_moyenne_globale_WER_CER(results)

    afficher_calcul_WER_CER_post_inference(wer_metric, cer_metric)
