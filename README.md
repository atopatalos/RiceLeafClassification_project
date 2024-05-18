# RiceLeafClassification

Classify Rice leaves.

## Installation and Setup

### Prerequisites

- Ensure your dataset is placed in the specified folder.
 > **Note**: The foldername `LabelledRice` has been changed to `DataLabelledRice`.
- Create an `mlruns` folder for MLflow to store its runs.

### Steps

1. **Clone the repository**:

    ```sh
    git clone https://github.com/Sudonuma/RiceLeafClassification.git
    ```

2. **Navigate to the cloned directory**:

    ```sh
    cd RiceLeafClassification
    ```

3. **Build your Docker image** (you can change `riceleaf` to any name you prefer for your Docker image):

    ```sh
    docker build -t riceleaf .
    ```

4. **Modify `start.sh`**:

    Ensure the `start.sh` file has the following content, replacing `{your_local_path}` with the actual path to your directories on your local machine:

    ```sh
    #!/bin/bash
    docker run -it -p 5000:5000 \
    -v {your_local_path}/mlruns:/app/mlruns \
    -v {your_local_path}/DataLabelledRice:/app/DataLabelledRice \
    riceleaf
    ```

    For example, if your local `mlruns` directory is located at `/home/user/Documents/mlruns` and your dataset is at `/home/user/Documents/DataLabelledRice`, your `start.sh` should look like this:

    ```sh
    #!/bin/bash
    docker run -it -p 5000:5000 \
    -v /home/user/Documents/mlruns:/app/mlruns \
    -v /home/user/Documents/DataLabelledRice:/app/DataLabelledRice \
    riceleaf
    ```

5. **Give execution permission to `start.sh`**:

    ```sh
    chmod +x start.sh
    ```

6. **Create the `mlruns` folder on your local machine** (if it doesn't exist):

    ```sh
    mkdir -p /home/user/Documents/mlruns
    ```

7. **Run `start.sh`**:

    ```sh
    ./start.sh
    ```

> **Note**: Your `mlruns` folder can become very large over time. If you have experiments you don't need, you can delete them from the folder. Alternatively, you can choose not to link an external volume for your MLflow runs, but be aware that your runs will not be saved. You can also modify your script to include a flag for whether or not to use MLflow.

### Re-running Training

Your Docker container will keep running after training, allowing you to see your experiment logs on MLflow. You can re-run your training with:

```sh
python src/train.py
```

This setup ensures your Docker environment is properly configured to run your training script and log experiments with MLflow, while also providing the flexibility to manage your MLflow runs as needed.

> **Note**: It's important to regularly clean up any unused Docker containers or images to free up disk space and maintain system performance.