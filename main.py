# main.py

import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Titanic ML Pipeline")

    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--train", action="store_true", help="Train Random Forest")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--save", action="store_true", help="Save trained model")
    parser.add_argument("--run_all", action="store_true", help="Run all steps")

    parser.add_argument("--data", type=str, default="train.csv", help="Path to dataset")
    parser.add_argument(
        "--model_out", type=str, default="model.pkl", help="Model output file"
    )
    parser.add_argument(
        "--model_in", type=str, default="model.pkl", help="Load model file"
    )

    args = parser.parse_args()

    X_train = X_val = y_train = y_val = None
    model = None

    # --- PREPARE DATA ---
    if args.prepare or args.run_all:
        print("Preparing data...")
        X_train, X_val, y_train, y_val = prepare_data(args.data)
        print("✔ Data ready.\n")

    # --- TRAIN MODEL ---
    if args.train or args.run_all:
        if X_train is None:
            X_train, X_val, y_train, y_val = prepare_data(args.data)

        print("Training model...")
        model = train_model(X_train, y_train)
        print("✔ Model trained.\n")

    # --- EVALUATE ---
    if args.evaluate or args.run_all:
        if model is None:
            print("Loading model...")
            model = load_model(args.model_in)

        if X_val is None or y_val is None:
            _, X_val, _, y_val = prepare_data(args.data)

        print("Evaluating model...")
        evaluate_model(model, X_val, y_val)
        print()

    # --- SAVE MODEL ---
    if args.save or args.run_all:
        if model is None:
            print("Error: No model to save.")
            return

        print(f"Saving model to {args.model_out}...")
        save_model(model, args.model_out)
        print("✔ Model saved.")


if __name__ == "__main__":
    main()
