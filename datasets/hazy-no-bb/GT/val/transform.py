import os

# Pfad zum Ordner (Punkt bedeutet aktuelles Verzeichnis)
folder_path = '.'


def rename_files(path):
    count = 0
    for filename in os.listdir(path):
        # Prüfen, ob die Datei auf _GT.png endet
        if filename.endswith("_GT.png"):
            # Neuen Namen generieren
            new_name = filename.replace("_GT.png", "_hazy.png")

            # Volle Pfade erstellen
            old_file = os.path.join(path, filename)
            new_file = os.path.join(path, new_name)

            # Umbenennen
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} -> {new_name}")
            count += 1

    print(f"\nFertig! {count} Dateien wurden umbenannt.")


if __name__ == "__main__":
    rename_files(folder_path)