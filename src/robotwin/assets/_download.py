import glob
import os
import zipfile

from huggingface_hub import snapshot_download

from robotwin.envs._GLOBAL_CONFIGS import ASSETS_PATH


def download_assets():
    print(f"Downloading assets to {ASSETS_PATH}...")
    if not os.path.exists(ASSETS_PATH):
        os.makedirs(ASSETS_PATH)
    
    # Map zip files to their expected directory names
    assets_map = {
        "background_texture.zip": "background_texture",
        "embodiments.zip": "embodiments",
        "objects.zip": "objects"
    }
    
    patterns_to_download = []
    
    for zip_name, dir_name in assets_map.items():
        dir_path = os.path.join(ASSETS_PATH, dir_name)
        if os.path.exists(dir_path):
            print(f"Asset '{dir_name}' already exists at {dir_path}. Skipping download.")
        else:
            patterns_to_download.append(zip_name)
            
    if not patterns_to_download:
        print("All assets are already present.")
        # We still run the update config path just in case
        print("Updating embodiment config paths...")
        update_embodiment_config_path(ASSETS_PATH)
        return

    print(f"Downloading: {patterns_to_download}")
    snapshot_download(
        repo_id="TianxingChen/RoboTwin2.0",
        allow_patterns=patterns_to_download,
        local_dir=ASSETS_PATH,
        repo_type="dataset",
        resume_download=True,
    )

    # Unzip files
    for zip_name in patterns_to_download:
        zip_path = os.path.join(ASSETS_PATH, zip_name)
        if os.path.exists(zip_path):
            print(f"Unzipping {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ASSETS_PATH)
            os.remove(zip_path)
        else:
            print(f"Warning: {zip_name} not found.")

    # Update embodiment config paths
    print("Updating embodiment config paths...")
    update_embodiment_config_path(ASSETS_PATH)

def update_embodiment_config_path(assets_path):
    BLUE = '\033[0;34m'
    YELLOW = '\033[0;33m'
    GREEN = '\033[0;32m'
    NC = '\033[0m'

    def print_color(message, color_code):
        print(f"{color_code}{message}{NC}")

    print_color(f"Assets path: {assets_path}", BLUE)

    # Check assets/embodiments
    if not os.path.isdir(os.path.join(assets_path, 'embodiments')):
        print_color("Warning: assets/embodiments directory not found", YELLOW)
        return

    # Export environment variable (for current process)
    os.environ['ASSETS_PATH'] = assets_path

    count_total = count_updated = count_error = 0

    # Find *_tmp.yml files
    print_color("Searching for configuration template files...", BLUE)
    pattern = os.path.join(assets_path, 'embodiments', '**', '*_tmp.yml')
    config_files = glob.glob(pattern, recursive=True)

    if not config_files:
        print_color("No *_tmp.yml files found", YELLOW)
        return

    print_color("Starting to process configuration files...", BLUE)
    for tmp_file in config_files:
        count_total += 1
        target_file = tmp_file.replace('_tmp.yml', '.yml')
        print(f"Processing [{count_total}]: {tmp_file} -> {target_file}")

        try:
            with open(tmp_file, 'r') as f:
                content = f.read()

            # Handle cases where template assumes ASSETS_PATH is root
            # e.g. ${ASSETS_PATH}/assets/embodiments -> .../assets/embodiments
            # assets_path ends with / usually
            assets_path_clean = assets_path.rstrip('/')
            
            new_content = content.replace('${ASSETS_PATH}/assets', assets_path_clean)
            new_content = new_content.replace('$ASSETS_PATH/assets', assets_path_clean)
            
            new_content = new_content.replace('${ASSETS_PATH}', assets_path)
            new_content = new_content.replace('$ASSETS_PATH', assets_path)

            with open(target_file, 'w') as f:
                f.write(new_content)

            print_color(f"  ✓ Successfully replaced ${{ASSETS_PATH}} -> {assets_path}", GREEN)
            count_updated += 1

        except Exception as e:
            print_color(f"  ✗ Replacement failed: {e}", YELLOW)
            count_error += 1

    print()
    print_color("Processing complete!", BLUE)
    print(f"Total processed: {count_total} files")
    print_color(f"Successfully updated: {count_updated} files", GREEN)

if __name__ == "__main__":
    download_assets()
