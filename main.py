
from inference import Model


def main():
    model = Model()
    print(model.generate_scene("A blue eyes fox under the table."))


if __name__ == "__main__":
    main()
