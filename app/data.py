from datasets import ClassLabel, Features, Value, load_dataset


def load_data(dataset_name: str = "bergr7/weakly_supervised_ag_news") -> tuple:
    # files
    labeled_data_files = {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }
    # features
    labeled_features = Features(
        {
            "text": Value("string"),
            "label": ClassLabel(
                num_classes=4,
                names=["World", "Sports", "Business", "Sci/Tech"],
            ),
        }
    )
    # load data
    labeled_dataset_train = load_dataset(
        dataset_name,
        data_files=labeled_data_files,
        features=labeled_features,
        split="train",
    )
    labeled_dataset_val = load_dataset(
        dataset_name,
        data_files=labeled_data_files,
        features=labeled_features,
        split="validation",
    )
    labeled_dataset_test = load_dataset(
        dataset_name,
        data_files=labeled_data_files,
        features=labeled_features,
        split="test",
    )

    return labeled_dataset_train, labeled_dataset_val, labeled_dataset_test


def load_unlabeled_data(
    dataset_name: str = "bergr7/weakly_supervised_ag_news",
):
    return load_dataset(
        dataset_name,
        data_files={"unlabeled": "unlabeled_train.csv"},
        features=Features({"text": Value("string")}),
        split="train",
    )
