import argparse
from datetime import datetime, timezone

#from daneel.detection import TransitModel
from daneel.parameters import Parameters
from daneel.detection.transit import transit
from daneel.detection.classifiers import RandomForestClassifier  # NEW (Task G)
from daneel.detection.data_process import load_data              # NEW (Task G)
from daneel.generation.gan_dream import dream   


def main():
    parser = argparse.ArgumentParser(description="Model and analyse exoplanet transits.")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    # CHANGED (Task G): -d now takes the name of the detection algorithm
    actions.add_argument("--dream", action="store_true", help="Train a GAN to generate synthetic transit light curves")
    actions.add_argument("-d", "--detection", choices=["rf", "cnn"], help="Run exoplanet detection with the chosen algorithm")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    parser.add_argument("-o", "--output", default="lc.png", help="Output image for --transit")

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.now(timezone.utc)

    input_pars = Parameters(args.input_file).params

    if args.transit:
        transit(input_pars['transit'], args.output)

        #transit = TransitModel(input_pars['transit'])
        #transit.plot_light_curve(args.output)
    elif args.dream:
            dream(input_pars['dream'])
    elif args.detection == "rf":
        # NEW (Task G): Random Forest detection, settings read from the YAML
        det = input_pars['detection']           # detection block of the YAML
        rf_pars = det['random_forest']          # Random Forest settings

        X_train, X_test, y_train, y_test, metadata, X_raw, X_err, scaler = load_data(
            csv_path=det['dataset_path'],
            n_bins=det['n_bins'],
            use_scaler=det['use_scaler'],
            samples_per_class=det['samples_per_class'],
        )

        rf = RandomForestClassifier(**rf_pars)
        rf.fit(X_train, y_train)
        rf.evaluate_with_optimal_threshold(X_test, y_test)
    elif args.detection == "cnn":
        parser.error("CNN detection is not implemented yet")
    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    finish = datetime.now(timezone.utc)
    print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()