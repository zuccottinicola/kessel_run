import argparse
from datetime import datetime, timezone

#from daneel.detection import TransitModel
from daneel.parameters import Parameters
from daneel.detection.transit import transit
from daneel.detection.classifiers import RandomForestClassifier  
from daneel.detection.classifiers import CNNClassifier
from daneel.detection.data_process import load_data              


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
        # Import GAN functions only when needed
        from daneel.generation.gan_dream import dream
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

        # Import CNN functions
        from daneel.detection.cnn_classifier import load_data as load_cnn_data
        from daneel.detection.cnn_classifier import plot_all
        import numpy as np

        # Read the CNN parameters from YAML
        cnn_pars = input_pars['cnn']

        # 1. Load and preprocess the dataset
        X_train, X_test, y_train, y_test, metadata_test, X_test_orig, X_err_test, scaler = load_cnn_data(
                csv_path=cnn_pars['csv_path'], n_bins=cnn_pars['nbins'])

        # 2. Initialize the CNN classifier
        cnn = CNNClassifier(n_bins=cnn_pars['nbins'],architecture=cnn_pars['model'])

        # 3. Train the CNN
        cnn.fit(X_train, y_train, X_test, y_test)

        # 4. Evaluate the CNN
        cnn.evaluate_with_optimal_threshold(X_test, y_test)

        # 5. Plot the results
        plot_all(
            y_test,
            cnn.y_pred,
            cnn.y_pred_proba,
            cnn.history,
            metadata_test,
            X_test,
            cnn.threshold,
            X_test_orig=X_test_orig,
            X_err_test=X_err_test,
            scaler=scaler
        )


        # 6. Save the model and the threshold
        cnn.model.save('tess_model_final.keras')
        np.save('optimal_threshold.npy', cnn.threshold)

    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    finish = datetime.now(timezone.utc)
    print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()