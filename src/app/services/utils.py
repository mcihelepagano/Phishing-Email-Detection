
import subprocess
import sys

from halo import Halo
from scipy import sparse

from app.services.datamanager import datamanager
from app.services.dl_trainer import DLTrainer
from app.services.inputs import ask_for_float, ask_for_integer, ask_for_range, ask_yes_no
from app.services.modelmanager import ModelManager
from app.services.printers.data_information_printer import DataInformationPrinter


def clear_console():
    subprocess.run('cls' if sys.platform == 'win32' else 'clear', shell=True)

async def load_the_dataset():
    spinner = Halo(text='Loading dataset...', spinner='dots')
    spinner.start()
    try:
        await datamanager.load_data()
        spinner.succeed('Dataset loaded successfully!')
    except Exception as e:
        spinner.fail(f'Failed to load dataset: {e}')
    return


async def get_info_about_dataset():
    print("--- Data Informations ---")
    if datamanager.df is None:
        print("Please load the dataset first.")
        return

    spinner = Halo(text="Analyzing dataset...", spinner="dots")
    spinner.start()

    stats = await datamanager.get_comprehensive_stats()
    data_printer = DataInformationPrinter(stats)

    spinner.succeed(text="Analysis complete")

    clear_console()

    # Display each section
    data_printer.print_basic_info()
    data_printer.print_target_distribution()
    data_printer.print_text_statistics()
    data_printer.print_email_specific_stats()
    data_printer.print_data_quality_report()


async def preprocess_data():
    """Preprocess the dataset by handling quality issues, feature engineering, and text vectorization."""
    if datamanager.df is None:
        print("Please load the dataset first.")
        return

    print("--- Data Preprocessing ---")

    threshold = ask_for_float("Drop columns with completeness < (percent)", default=50.0, min=0.0, max=100.0)

    drop_constants = ask_yes_no("Drop all constant columns?", default=True)

    spinner = Halo(text='Handling quality issues...', spinner='dots')
    spinner.start()
    try:
        await datamanager.handle_quality_issues(drop_constants=drop_constants, threshold=threshold)
        spinner.succeed('Data quality issues treated!')
    except Exception as e:
        spinner.fail(f'Handling quality issues failed: {e}')

    print("--- Feature Engineering ---")
    create_features = ask_yes_no(
        "Create ML feature columns (text_length, flags, sender/recipient domains + one-hot)?", default=True
    )
    if create_features:
        top_k = ask_for_integer("Max number of top domains to one-hot encode", default=50, min=1)
        fe_spinner = Halo(text='Creating ML feature columns...', spinner='dots')
        fe_spinner.start()
        try:
            await datamanager.run_feature_engineering(top_k_domains=top_k)
            fe_spinner.succeed('Feature creation complete!')
        except Exception as e:
            fe_spinner.fail(f'Feature creation failed: {e}')

    do_vectorize = ask_yes_no("Vectorize text for ML now?", default=True)
    if not do_vectorize:
        return

    print("--- Text Vectorization Parameters ---")
    ngram_range = ask_for_range("N-gram range (min-max)", default=(1, 2))
    max_features = ask_for_integer("Max features for vectorizer", default=10000, min=100)

    spinner = Halo(text='Vectorizing text...', spinner='dots')
    spinner.start()
    try:
        X = await datamanager.run_vectorization(ngram_range=ngram_range, max_features=max_features)
        spinner.succeed('Vectorization complete!')
        # Report shape if available
        try:
            print(f"Feature matrix shape: {getattr(X, 'shape', None)}")
        except Exception:
            pass
    except Exception as e:
        spinner.fail(f'Vectorization failed: {e}')
    return


modelmanager = ModelManager(datamanager)
dl_trainer = DLTrainer(datamanager)


async def train_model_menu():
    print("Choose a model to train:")
    print("1. Random Forest")
    print("2. Stochastic Gradient Descent (SGD)")
    print("3. Linear SVM")
    print("4. Decision Tree")

    choice = int(input("Enter choice: "))
    if choice == 1:
        modelmanager.train_model("random_forest")
    elif choice == 2:
        modelmanager.train_model("sgd")
    elif choice == 3:
        modelmanager.train_model("linear_svm")
    elif choice == 4:
        modelmanager.train_model("decision_tree")
    else:
        print("Invalid choice.")
        return

    save = input("Do you want to save this model? (y/n): ").lower()
    if save == "y":
        modelmanager.save_model()

async def train_dl_model_menu():
    if datamanager.df is None:
        print("Please load the dataset first.")
        return

    print("Deep Learning options:")
    print("1. Train lightweight TextCNN (cross-entropy)")
    print("2. Train lightweight TextCNN (focal loss + optional GA threshold)")
    print("3. Train DistilBERT")
    print("4. Evaluate saved DistilBERT model")
    print("5. Evaluate saved lightweight DL model")

    choice = ask_for_integer("Enter choice", default=1, min=1, max=5)
    if choice in (1, 2):
        epochs = ask_for_integer("Epochs", default=4, min=1)
        batch_size = ask_for_integer("Batch size", default=32, min=4)
        lr = ask_for_float("Learning rate", default=1e-3, min=1e-5, max=5e-2)
        max_length = ask_for_integer("Max tokens per email", default=200, min=50, max=400)
        max_vocab = ask_for_integer("Max vocabulary size", default=20000, min=2000, max=50000)
        sample_limit = ask_for_integer("Max samples to use (0 = use all)", default=80000, min=0)
        scoring = "cross_entropy" if choice == 1 else "focal"
        use_ga = False
        focal_gamma = 2.0
        if choice == 2:
            focal_gamma = ask_for_float("Focal gamma", default=2.0, min=0.5, max=5.0)
            use_ga = ask_yes_no("Run GA to tune decision threshold after training? (binary only)", default=True)
        print(f"Using device: {dl_trainer.device}")
        dl_trainer.train_text_cnn(
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            max_length=max_length,
            max_vocab_size=max_vocab,
            train_sample_limit=sample_limit,
            scoring=scoring,
            focal_gamma=focal_gamma,
            use_genetic_threshold=use_ga
        )
    elif choice == 3:
        epochs = ask_for_integer("Epochs", default=2, min=1)
        batch_size = ask_for_integer("Batch size", default=8, min=2)
        lr = ask_for_float("Learning rate", default=5e-5, min=1e-6, max=1e-3)
        max_length = ask_for_integer("Max sequence length", default=256, min=64, max=512)
        sample_limit = ask_for_integer("Max train samples (0 = use all)", default=0, min=0)
        print(f"Using device: {dl_trainer.device}")
        dl_trainer.fine_tune_distilbert(
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            max_length=max_length,
            train_sample_limit=sample_limit
        )
    elif choice == 4:
        batch_size = ask_for_integer("Batch size", default=8, min=2)
        dl_trainer.evaluate_saved_distilbert(batch_size=batch_size)
    elif choice == 5:
        batch_size = ask_for_integer("Batch size", default=32, min=4)
        dl_trainer.evaluate_saved_model(batch_size=batch_size)
    else:
        print("Invalid choice.")

async def load_saved_model():
    model_name = input("Enter the model name (e.g., RandomForest or SGDClassifier): ")
    modelmanager.load_model(model_name)

async def continue_training_model():
    evaluate = ask_yes_no("Evaluate with a holdout split before final fit?", default=True)
    test_size = 0.2
    if evaluate:
        test_size = ask_for_float("Holdout fraction", default=0.2, min=0.05, max=0.5)
    modelmanager.continue_training(evaluate=evaluate, test_size=test_size)
    save = input("Save the updated model? (y/n): ").lower()
    if save == "y":
        modelmanager.save_model()

async def export_test_split():
    """Export cached test split features/labels for external analysis (e.g., Colab)."""
    if datamanager.df is None or getattr(datamanager, "X_processed", None) is None:
        print("Please load, preprocess, and vectorize the dataset first.")
        return
    try:
        mm = ModelManager(datamanager)
        X, y = mm.get_features_and_labels()
        splits = datamanager.ensure_split()
        test_idx = splits["test"]
        X_test = X[test_idx]
        y_test = y.iloc[test_idx]
        sparse.save_npz("X_test.npz", X_test)
        y_test.to_csv("y_test.csv", index=False)
        print("Saved test split to X_test.npz and y_test.csv in the current directory.")
    except Exception as e:
        print(f"Export failed: {e}")


async def menu():
    options = {
        1: ("Load the dataset", load_the_dataset),
        2: ("Get information about the dataset", get_info_about_dataset),
        3: ("Preprocess data", preprocess_data),
        4: ("Train a classical ML model", train_model_menu),
        5: ("Load a saved model", load_saved_model),
        6: ("Continue training a loaded model", continue_training_model),
        7: ("Train a DL model", train_dl_model_menu),
        8: ("Export test split (X_test, y_test)", export_test_split)
    }

    while True:
        clear_console()
        print("--- Phishing Email Detection ---")
        for key, (desc, _) in options.items():
            print(f"[{key}] {desc}")
        print("[0] Exit")

        choice = ask_for_integer("Choose", default=None, min=0, max=len(options))
        if choice is None:
            continue

        if choice == 0:
            print("Bye!")
            break

        _, action = options[choice]
        clear_console()
        await action()
        input("Press Enter to continue...")
