
from inference import Model


def main():
    model = Model()
    print(model.generate_scene("A blue eyes dog on a wooden table"))


if __name__ == "__main__":
    main()
