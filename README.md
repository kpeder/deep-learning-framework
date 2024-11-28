# Python Deep Learning Framework
A Python framework for creating, training, testing and tuning machine learning models.

## Objectives
1. Demonstrate experimental patterns related to the testing and tuning of machine learning models
1. Provide methods for and examples of repeatable, quantitative measurement of model performance
1. Leverage machine learning models to automate the testing and tuning of machine learning models

## Prerequisites
Some elements of the framework may require external components to be installed and running.

1. Docker and Minikube or an alternative local conformant Kubernetes deployment.
1. Kubeflow installed and running in the Kubernetes cluster.
1. A Kuberenetes secret configured for a GCP service account key using the instructions found [here](https://googlecloudplatform.github.io/kubeflow-gke-docs/docs/pipelines/authentication-pipelines/#google-service-account-keys-stored-as-kubernetes-secrets).

## Configuration
The framework is tested on Ubuntu 24.04 with Python 3.10.15 and 3.12.3.

NOTE: Tensorflow Extended pipelines do not yet support Python 3.12, so Python 3.10 is recommended.

1. Install required system packages (Ubuntu-specific):

    ```$ sudo apt-get install -y build-essential git pre-commit python3-pip python3-dev```

1. If compiling Python from source, additional source libraries may be needed:

    ```$ sudo apt-get install -y libbz2-dev libffi-dev liblzma-dev libsnappy-dev libsqlite3-dev libssl-dev```

1. Clone this repository:

    ```$ git clone https://github.com/kpeder/deep-learning-framework.git```

1. Optionally, use a virtual environment such as [venv](https://realpython.com/python-virtual-environments-a-primer/#create-it), [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html#quick-start) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#creating-environments). If you want to use a virtual environment, make sure that it is created and activated before continuing.

1. Install required Python packages:

    ```$ make install```

1. Run package tests:

    ```$ make test```

## Running the Examples
Make sure you've set your PYTHONPATH:

```$ export PYTHONPATH=`pwd`/src```

Run the examples (some are long-running!):

1. Create and train a static model.

    ```$ python3 examples/mnist.py```

1. Create and train a Hypermodel.

    ```$ python3 examples/tuner.py```

1. Create and train a Hypermodel using multiple threads.

    ```$ python3 examples/mptuner.py```

1. Get the best Hyperparameters across trial runs.

    ```$ python3 examples/tquery.py --project-prefix ran --top-trials 3```

1. Create a local TFX pipeline.

    ```$ python3 examples/pipeline.py```

## Contributing
Feel free to contribute to this framework via [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).
  - [ ] All checks and tests must pass using ```make pre-commit; make test```
  - [ ] The contribution must align with architecture [decisions](./docs/decisions/index.md)
  - [ ] The contribution should build toward the project's [objectives](README.md#objectives)

[Issues](https://github.com/kpeder/deep-learning-framework/issues) may be submitted and might be considered for implementation, but there are no guarantees and no service commitment is in place.

## License
This repository is formally [Unlicensed](./LICENSE.md). Do what you will with any copy of it.
