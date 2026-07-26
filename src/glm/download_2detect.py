from pathlib import Path
import subprocess

def run_cmd(cmd, verbose=True, *args, **kwargs):
    if verbose:
        print(cmd)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()

    if process.returncode:
        print("Error encountered in commad line call:")
        raise RuntimeError(std_err.strip())
    if verbose:
        print(std_out.strip(), std_err)
    return std_out.strip()

def unzip_file(zipped_file_path: Path, unzipped_file_path: Path) -> None:
    """Unzip file. First tries unzip command then 7zip if unzip fails"""
    try:
        print(f"Unzipping {zipped_file_path} files with unzip...")
        bash_command = f"unzip {zipped_file_path} -d {unzipped_file_path}"
        run_cmd(bash_command)
    except RuntimeError:
        try:
            print("Unzipping with Unzip failed. Unzipping with 7zip")
            bash_command = f"7z x {zipped_file_path} -o {unzipped_file_path} -y"
            run_cmd(bash_command)
        except RuntimeError:
            raise RuntimeError("Both unzip and 7zip failed for unzipping")

    print("Extraction done!")
    zipped_file_path.unlink()
    print(f"{zipped_file_path} deleted.")


def download_file(placeholder_url: str, file_path: Path) -> None:
    """
    Downloads file from placeholder_url. Tries first with wget, then with curl.

    """
    if not file_path.is_file():
        try:
            print(f"Downloading {placeholder_url} with wget...")
            bash_command = (
                f"wget {placeholder_url} -P {file_path.parent} -O {file_path.stem}"
            )
            run_cmd(bash_command)

        except RuntimeError:
            try:
                print("Downloading with wget failed. Downloading with curl...")
                bash_command = f"curl {placeholder_url} > {file_path}"
                run_cmd(bash_command)
            except RuntimeError:
                raise RuntimeError("curl and wget failed, cannot download")

        print("Dowload DONE!")
    else:
        print(f"{file_path.stem} already exists in {file_path.parent}, passing")


storage_path = Path('/media/Store-HDD/emilien_temp/raw/2detect')
file_ids = {3723295: [i for i in range(0, 7)], 4121926: [i for i in range(7, 10)]}

data_ids = {
    ### Sinogram
    "2DeteCT_slices1-1000": 8014758,
    "2DeteCT_slices1001-2000": 8014766,
    "2DeteCT_slices2001-3000": 8014787,
    "2DeteCT_slices3001-4000": 8014829,
    "2DeteCT_slices4001-5000": 8014874,
    "2DeteCT_slicesOOD": 8014907,
    ### Reconstructions
    "2DeteCT_slices1-1000_RecSeg": 8017583,
    "2DeteCT_slices1001-2000_RecSeg": 8017604,
    "2DeteCT_slices2001-3000_RecSeg": 8017612,
    "2DeteCT_slices3001-4000_RecSeg": 8017618,
    "2DeteCT_slices4001-5000_RecSeg": 8017624,
    "2DeteCT_slicesOOD_RecSeg": 8017653,
}  # recon, seg

if __name__ == "__main__":
    for series_name, zenodo_id in data_ids.items():
        placeholder_url = (
            f"https://zenodo.org/records/{zenodo_id}/files/{series_name}.zip"
        )
        zipped_file_path = storage_path.joinpath(f"{series_name}.zip")
        unzipped_file_path = storage_path.joinpath(f"{series_name}")

        download_file(placeholder_url, zipped_file_path)
        unzip_file(zipped_file_path, unzipped_file_path)

    print("done!")