
from inference import Model


def main():
    model = Model(experiment_name="nanogpt_v0.4_seed42")
    print(model.generate_scene("A dog on a wooden table"))


if __name__ == "__main__":
    main()
