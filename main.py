
from inference import Model


def main():
    model = Model(experiment_name="nanogpt_v0.3_seed42")
    print(model.generate_scene("The cup is ceramic and blue. Place it on a wooden table."))


if __name__ == "__main__":
    main()
