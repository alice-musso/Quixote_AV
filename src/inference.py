import warnings

warnings.filterwarnings("ignore")

def main():
    from config import ModelConfig
    from experiment_runner import QuixoteInferenceExperiment

    config = ModelConfig.from_args()
    experiment = QuixoteInferenceExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
