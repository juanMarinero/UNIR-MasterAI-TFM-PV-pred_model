<!-- vim: set ts=2 sts=2 sw=2 expandtab tw=0 foldcolumn=5 foldmethod=expr : -->


# How to use

```sh
# Clone the repository
repo="UNIR-MasterAI-TFM-PV-pred_model"
git clone https://github.com/juanMarinero/$repo.git
cd $repo

# Create a 🐍 Python virtual environment (version 3.13.2)
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install jupyter
# ...

# Install requirements
pip install -r requirements.txt
```

Downlaod the dataset via `cd scripts && python build_pvdt_dataset.py` (to run `process_all_months` function).

## Run the notebook

```sh
jupyter lab
```

Previous to commit convert to HTML and check it (for easier inspection):

```sh
jt -r # Close the jupyter session and run `jt -r` to restore default theme
jupyter nbconvert --to html --HTMLExporter.theme=light foo.ipynb # and open the HTML in browser
```


## Run a script

```sh
python source/script.py
```


# Git tips

Run `git status` to check the status of the repository. 

Run `git diff` to check the changes.

Run `git log` to check the commit history.

❗ Run `git pull` to update the repository. ❗

To commit run `git add foo.py` and `git commit -m "commit message"`.
