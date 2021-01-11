# Template for Deeptech Projects

> This serves as a simple template for deeptech projects.

**Abstract** - Write some short info what your project is about (1-2 paragraphs).

![Teaser Image](teaser.png)

## Installation

Install all dependencies by running the following command in your conda environment.

```bash
pip install -r requirements.txt
```

## Running

If the project was created via deeptech_tools you can simply run it from your vs code debug terminal.

If you want to run the program from the commandline use this command to train (but replace template with the folder name where the code lives):
```bash
python -m template.config.train --mode=train --input=input --output=results
```

Similarily for testing use this command (but replace template with the folder name where the code lives):
```bash
python -m template.config.test --mode=test --input=input --output=results
```

## Licenses

The code provided by the template falls under the MIT License. You might be able to remove this license note, once you have removed the code in the template and replaced it with your own. Then put the new appropriate licenses in this paragraph and update the LICENSE file.
