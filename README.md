# Reveal-AI

A study of SOTA models and their ability to detect AI-generated imagery when faced with common transformations like cropping and resizing.


## Setup Instructions

### Using UV Package Manager

1. **Install** `uv` by running the following command in your terminal:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

Read more about installing `uv` [here](https://docs.astral.sh/uv/getting-started/installation/).


2. **Create** a virtual environment named .venv in the project root by running:
   
   ```bash
   uv venv .venv --python 3.12
   ```

   This will also install the specified Python version if it's not already available on your system.

   **Activate** the virtual environment by running the command provided upon successful creation.


3. If a 'pyproject.toml' file already exists in the project root, run:

   ```bash
   uv sync
   ```

   This will install the main dependencies of the project. To install all optional dependency groups (like dev, test, docs), run:

   ```bash
   uv sync --all-extras --all-groups
   ```

4. To install additional libraries:
   
    ```bash
    uv add <package-name>
    ```

    For a specific version:
    
    ```bash
    uv add '<package-name>==1.2.3'
    ```

    To add to a specific dependency group:
    
    ```bash
    uv add <package-name> --group <group-name>
    ```

5. To uninstall libraries:
   
    ```bash
    uv remove <package-name>
    ```

6. Run familiar `pip` commands using `uv`:
   
    ```bash
    uv pip install <package-name>
    uv pip uninstall <package-name>
    uv pip list
    ```

    **NOTE**: These commands will operate within the `.venv` virtual environment, but will not update the `pyproject.toml` file. Use `uv add` and `uv remove` to manage dependencies in the project configuration.
