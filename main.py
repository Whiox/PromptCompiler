
from inference import Model


def main():
    model = Model(experiment_name="nanogpt_v0.4_2_seed42")
    print(model.generate_scene("Show the red ceramic cup next to the blue metal box."))


if __name__ == "__main__":
    main()
