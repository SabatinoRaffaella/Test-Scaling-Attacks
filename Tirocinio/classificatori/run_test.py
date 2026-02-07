import os
from enum import Enum

import torch
import os, sys
# Path to /content
PROJECT_ROOT = "/content/Tirocinio"
sys.path.insert(0, PROJECT_ROOT)

from attack.factory import attack_factory, NormalizedModel
from training import evaluate, load_pretrained_model, finetune_classifier
from training.classifier_utilities import make_model, get_num_classes
from training.eval_utils import evaluate_jpeg_recovery
from utils.config import PROJECT_ROOT, cfg, ModelConfig, AttackConfig, DefenseConfig
from utils.dataset_utilities import build_dataloaders
from utils.log_results import postprocess_results, save_jpeg_recovery_csv, plot_jpeg_recovery, save_adv_samples
from utils.image_utilities import denormalize_tensor, normalize_tensor

import json
from tqdm import tqdm
def quality_batchifier(ql_rank):
      '''
    Funzione per selezionare una lista di valori compresi
    in un intervallo determinato da ql_rank.
    OPZIONI sono LOW, MEDIUM_LOW e MEDIUM.
    \u25CFLOW: Include valori nell'intervallo 10-20 (inclusi).\n
    \u25CFMEDIUM_LOW: Include valori nell'intervallo 21-40 (inclusi).\n
    \u25CFMEDIUM: Include valori nell'intervallo 41-50 (inclusi).\n
    :param ql_rank: Enum qualità di compressione.
    :return: Restituisce una lista di valori nel range scelto.
    '''
    match ql_rank:
        case Quality_RANK.LOW: return list(range(10, 21))
        case Quality_RANK.MEDIUM_LOW: return list(range(21, 41))
        case Quality_RANK.MEDIUM: return list(range(40, 61))
        case _: return []  # Default case for any invalid input


class Quality_RANK(Enum):
    """Classe Enum per i Range della qualità di compressione\n
        LOW (10-20)\n
        MEDIUM_LOW (21-40)\n
        MEDIUM (40-60)"""
    LOW = 1
    MEDIUM_LOW = 2
    MEDIUM= 3

def first_tensor(obj):
    """Restituisce il primo torch.Tensor dentro obj (tupla/lista)."""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = first_tensor(v)
            if t is not None:
                return t
    return None

def run_grid(
    models,
    quality_batch,
    loader_fn: callable,
    device: torch.device,
    cfg,
    out_dir="attack_results",
    max_batches=None,
    save_examples=True,
):

    all_results = {}
    # 0) Salva i risultati JPEG per-batch per fare la media.
    jpeg_results_accum = {q: [] for q in quality_batch}

    # 1) Itero sui modelli:
    for model_name, (model_cfg, base_model, preprocess) in models.items():
        try:
            print("DEBUG pretrained =", model_cfg.pretrained)

            # 2) Prepara i dati prima di elaborarli
            # Si assicura che le immagini rispettino le dimensioni
            # Standard di ImgNET
            # ---3) Prepara i dataset di training e evaluation: ---
            train_loader, val_loader = loader_fn(base_model, preprocess)

            # 4) Esegue il Fine-Tuning del modello su TinyImgNET se il modello non è salvato.
            # 4-1) Altrimenti lo carica.
            base_model = finetune_classifier(
                base_model,
                train_loader,
                val_loader,
                device,
                epochs=5,
                lr=1e-3,
                freeze_backbone_flag=True,
                model_name=model_cfg.model_name,
                out_dir=os.path.join(out_dir, "checkpoints")
            )
            print("Detected num_classes:", get_num_classes(base_model))

            # 5) Fare la eval per la clean accuracy
            clean_accuracy = evaluate(base_model,val_loader,device)
            print(clean_accuracy)

            #6) Wrappiamo il modello nella classe NormalizedModel in modo tale
            # da specificare che questo si aspetta input normalizzati.
            model_aa = NormalizedModel(base_model, normalize_tensor).to(device)
            model_aa.eval()

            # 5) Istanzio l'oggetto per fare l'attacco
            attack = attack_factory("PGD",model_aa,device)
            print ({attack})

            # 6) Recupero (dal Dataloader) le immagini attaccate (inputs e etichette)
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Attacking {model_cfg.model_name}", total=max_batches)):
                if max_batches is not None and batch_idx >= max_batches:
                    break

               # inputs are normalized (because val_loader uses preprocess). Convert to [0,1]
                inputs_raw = batch[0]
                labels_raw = batch[1]

                 # If inputs_raw is nested e.g. ((img, aux), ...), get the first tensor
                inputs = first_tensor(inputs_raw)
                labels = first_tensor(labels_raw)

                inputs = inputs.to(device)
                labels = labels.to(device).long()
                # 1) Denormalizza -> Porta il valore dei pixel a valori in [0,1] (pixel space)
                inputs_unnorm = denormalize_tensor(inputs)
                inputs_unnorm = inputs_unnorm.clamp(0.0, 1.0)

                # 2) Attacco viene eseguito sulle immagini UNNORMALIZED (torchattacks expects [0,1])
                adv_unnorm = attack(inputs_unnorm, labels)

                # 3) Clamp to valid range
                adv_unnorm = adv_unnorm.clamp(0.0, 1.0)

                # 4) Convert back to normalized input for model inference
                adv_norm = normalize_tensor(adv_unnorm)

                # --- adversarial ---
                with torch.no_grad():
                    adv_logits = base_model(adv_norm)
                    adv_preds = adv_logits.argmax(dim=1)
                    adv_correct_mask = adv_preds == labels
                    adv_accuracy = adv_correct_mask.float().mean().item()

                # --- clean ---
                with torch.no_grad():
                    clean_logits = base_model(inputs)
                    clean_preds = clean_logits.argmax(dim=1)
                    clean_correct_mask = clean_preds == labels

                # Now adv_norm can be fed to the model for direct predictions,
                # and adv_unnorm (pixel-space) can be used for JPEG recovery evaluation
                attacked_images = adv_unnorm

              #  print("DEBUG labels AFTER ATTACK:", labels, type(labels))
                # 7) Salvo qualche sample di adversarial example
                save_adv_samples(out_dir, model_cfg, attacked_images, inputs)

                # LOW ACCURACY
                # NO ESECUZIONE
                # 8) Valuto a vari livelli di compressione
                batch_results = evaluate_jpeg_recovery(
                    base_model,
                    attacked_images,
                    labels,
                    quality_batch,
                    normalize_tensor,
                    clean_correct_mask,
                    adv_correct_mask,
                    clean_images=inputs_unnorm,
                    save_dir=os.path.join(out_dir, model_cfg.model_name, "jpeg_correct"),
                    prefix=f"batch{batch_idx}"
                )
                # accumulate for later averaging
                for q, acc in batch_results.items():
                    jpeg_results_accum[q].append(acc)

                # Calcola la media per plottare grafico accuracy/compressione
            jpeg_mean = {
                q: (sum(accs) / len(accs) if accs else 0.0)
                for q, accs in jpeg_results_accum.items()
            }

            # Salva clean accuracy e media
            all_results[model_cfg.model_name] = {
                "adv_accuracy": adv_accuracy,
                "clean_accuracy": clean_accuracy,
                "jpeg_mean": jpeg_mean,
            }

        except Exception as e:
            print (f"[ERROR] Exception with {model_cfg.model_name}: {e}")
    return all_results

def main():
    import gc, torch   # always available now

    #Setta il device iniziale per la configurazione (CUDA)
    device = cfg.run.device
    print("Sto caricando il dataset...")

    #Carica i DataLoader:
    def loader_fn(model, preprocess):
        return build_dataloaders(
            model,
            data_root=cfg.data.data_root,
            preprocess = preprocess,
            debug=False
        )

    #Selezione rango qualità da testare
    selected_rank = Quality_RANK.MEDIUM_LOW

    # 🔥 CARICO TUTTI I MODELLI QUI — prima di run_grid()
    # ---2) Usa la factory timm per creare i modelli ---
    import timm
    print("🔽 Preloading model weights...")
    built_models = {}
    for mcfg in cfg.models:
        print(f"Building {mcfg.model_name} with {mcfg.num_classes} classes...")
        model, preprocess, weights = load_pretrained_model(
            mcfg.model_name,
            device=device
        )
        built_models[mcfg.model_name] = (mcfg, model.to(device), preprocess)
        print(f"DEBUG pretrained = {mcfg.pretrained}")
    # Recupera tutti i dati per gli esperimenti dal file di configurazione
    # e crea la cartella per gli output.
    out_dir = PROJECT_ROOT+"/"+cfg.run.out_dir
    try:
        results = run_grid(
            models=built_models,
            quality_batch= quality_batchifier(selected_rank),
            loader_fn=loader_fn,
            device=device,
            cfg=cfg,
            out_dir=out_dir,
            max_batches=cfg.run.max_batches,
            save_examples=cfg.run.save_examples
        )
        # NEW: run plotting/analysis separately
        print("All results:", results)

        # Itero sui risultati finali per loggarli/salvarli
        for model_name, data in results.items():
            jpeg_mean = data["jpeg_mean"]
            clean_acc = data["clean_accuracy"]
            adv_acc = data["adv_accuracy"]
            model_dir = f"results/{model_name}"
            os.makedirs(model_dir, exist_ok=True)
            metrics = {
                "clean_accuracy": clean_acc,
                "adv_accuracy": adv_acc,
                "jpeg_mean_accuracy": jpeg_mean,
            }
            # Clean accuracy
            with open(f"{model_dir}/metrics.json", "w") as f:
              json.dump(metrics, f, indent=4)

            # CSV
            save_jpeg_recovery_csv(jpeg_mean, f"{model_dir}/jpeg_recovery.csv")

            # Plot
            plot_jpeg_recovery(jpeg_mean, f"{model_dir}/jpeg_recovery.png")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()