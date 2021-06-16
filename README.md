# Step 1: Install Miniconda
```bash
install_miniconda_mac () {
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ~/miniconda3/bin/conda init zsh
}
install_miniconda_linux() {
    mkdir -p ~/.miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/.miniconda3/miniconda.sh
    bash ~/.miniconda3/miniconda.sh -b -u -p ~/.miniconda3
    rm -rf ~/.miniconda3/miniconda.sh
    ~/.miniconda3/bin/conda init bash
    ~/.miniconda3/bin/conda init zsh
}
install_miniconda_mac
```
# Step 2: Create conda environment
```bash
git clone https://github.com/rasoolianbehnam/mri_project.git

cd mri_project

conda env create --force

conda activate mri_project

jupyter notebook
```
