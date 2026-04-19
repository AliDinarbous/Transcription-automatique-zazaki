from transformers import WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from datasets import load_dataset, DatasetDict, Audio, Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import pandas as pd
import evaluate
import torch
import os


"""
NB: La methode load_dataset de huggingface ne marche plus pour les nouveaux datasets de common voice, Alors on telecharge le dataset
en local, on le convertis en dataframe pandas, puis on le convertit en dataset huggingface et on fait le reechantillonage des datasets 
"""
def load_local_dataset(PATH):
    # Declarer le dataset au format de huggingface (DatasetDict)
    common_voice = DatasetDict()
    
    # chemin absolu vers le datsets 
    directory = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets"
    #chemin vers le fichier audio
    PATH_TO_AUDIO = "/info/raid-etu/m1/s2506992/projet-m1-asr/datasets/" + PATH + "clips/"

    # Creer les chemins dynamiques
    tsv_path_train = os.path.join(directory, PATH, "train.tsv")
    tsv_path_dev = os.path.join(directory, "..", "datasets", PATH, "dev.tsv")
    tsv_path_test = os.path.join(directory, "..", "datasets", PATH, "test.tsv")

    # Convertir en dataframe oandas
    df_train = pd.read_csv(tsv_path_train, delimiter='\t')
    df_dev = pd.read_csv(tsv_path_dev, delimiter='\t')
    df_test = pd.read_csv(tsv_path_test, delimiter='\t')

    # Convertir en dataset huggingface
    common_voice["train"] = Dataset.from_pandas(df_train)
    common_voice["dev"] = Dataset.from_pandas(df_dev)
    common_voice["test"] = Dataset.from_pandas(df_test)

    # Creer la colonne audio pour les trois corpus, elle contient le chemin complet vers l'audio
    common_voice["train"] = common_voice["train"].map( lambda x: {"audio": PATH_TO_AUDIO + x["path"]})
    common_voice["dev"] = common_voice["dev"].map( lambda x: {"audio": PATH_TO_AUDIO + x["path"]})
    common_voice["test"] = common_voice["test"].map( lambda x: {"audio": PATH_TO_AUDIO + x["path"]})

    # On fait le Reechantillonnage des fichiers audio en 16000khz (les audios de common voice sont en 48000khz par defaut) 
    common_voice["train"] = common_voice["train"].cast_column("audio", Audio(sampling_rate=16_000))
    common_voice["dev"] = common_voice["dev"].cast_column("audio", Audio(sampling_rate=16_000))
    common_voice["test"] = common_voice["test"].cast_column("audio", Audio(sampling_rate=16_000))
    
    return common_voice

"""
cette methode sert a eliminer tout les colonnes non necessaires pour l'apprentissage, on garde une nouvelle colonne: inputfeatures; elle contient un tableau
numpy qui represente les caracteres acoustiques de l'audio, et une colonne label; elle contient la label dans le format Token
"""
def prepare_dataset(batch, feature_extractor, tokenizer):
    # recuperer les colonnes 'audio'
    audio = batch["audio"]

    # on cree une nouvelle colonne 'input_features' qui contiendra les tableaux numpy qui represente le fichier audio 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # on transforme le text en token numerique
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    return batch


"""
Un Data Collator sert a preparer le bacth avant de l'envoyer au model, dans notre cas il va servir pour faire 3 choses:
- etape 1: on transforme la colonne 'input features' qui contient les tableaux numpy en tensor pytorch
- etape 2: on unifie la taille des tokens numerique (qui represente les labels) en ajoutant des <pad> qui seront apres remplace par des -100 pour ne pas biaiser 
  les metrics d'evaluation
- etape 3: on supprime le token <s> qui indique le debut de sequence
"""

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # etape 1: 
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # etape 2
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # etape 3
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        return batch
        
# Charger la  metric
metric = evaluate.load("wer")

# Defintion des Hyperparametre de l'entrainement
training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join("..", "results","whisperBaseEnglish"),  # repertoire enregistrement des chekcpoints
    do_train=True,                      
    do_eval=True,  
    per_device_train_batch_size=4,        # petit batch par GPU
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,        # accumulation de gradient ~32
    learning_rate=1e-2,                    #0.01
    warmup_steps=100,                      # warmup pour stabiliser debut d'apprentissage
    num_train_epochs=10,                   # 10 epochs complètes
    lr_scheduler_type="cosine",           # scheduler cosinus
    generation_num_beams=5,
    generation_max_length=225,
    gradient_checkpointing=False,         # économise la mémoire
    fp16=True,                            
    eval_strategy="steps",
    eval_steps=25,
    save_steps=50,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    remove_unused_columns=False
)

def whisper_finetune (PATH):

    # importer le processor 
    processor = WhisperProcessor.from_pretrained("openai/whisper-base",language="turkish", task="transcribe")
    # importer l'extracteur de features
    feature_extractor = processor.feature_extractor
    # importer le otkenizer
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", language="turkish", task="transcribe")
    # initier le model
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # on recupere le dataset
    common_voice = load_local_dataset(PATH)
    
    # le dataset avec deux colonnes, input_features: spectogramme mel et labels: les tokens numerique des etiquetes
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], fn_kwargs={"feature_extractor": feature_extractor,  "tokenizer": tokenizer})

    # Initialiser le datacollator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
    )

    """
    elle prends les predictions, elle les compare les predictions et les labels, et retourne la word_error_rate
    """
    def compute_metrics(pred):

        normalizer = BasicTextNormalizer()
    
        # on recupere les predictions
        pred_ids = pred.predictions
    
        # on recupere les labels 
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # on decode (on convertit les tokens numerique en text) les deux, pred_ids et label_ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalizer(s) for s in pred_str]
        label_str = [normalizer(s) for s in label_str]

        # calcul du taux d'erreur
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Initialiser le trainer
    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    )
    
    # lancer l'entrainement
    trainer.train()
    
if __name__ == "__main__":
    PATH = 'zazaki/'
    whisper_finetune(PATH)