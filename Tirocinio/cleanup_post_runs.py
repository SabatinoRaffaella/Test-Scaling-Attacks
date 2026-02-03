import os, shutil, fnmatch
def clear_runs(attack_name):
    base = "Tirocinio/attack_results"

    for folder in os.listdir(base):
        if attack_name in folder:
            attack_path = os.path.join(base, folder)

            for sub in os.listdir(attack_path):
                sub_path = os.path.join(attack_path, sub)

                # Skip checkpoint folder
                if sub == "checkpoints":
                    print("Skipping checkpoints:", sub_path)
                    continue

                # Delete only run directories
                if os.path.isdir(sub_path):
                    shutil.rmtree(sub_path)
                    print("Deleted:", sub_path)
clear_runs("efficientnet_b0")
clear_runs("resnet18")