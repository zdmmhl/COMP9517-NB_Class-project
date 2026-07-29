"""Compatibility entry point for checkpoint-only deep model evaluation."""

from _bootstrap import load_main

main = load_main("training.train_deep")


if __name__ == "__main__":
    main()
