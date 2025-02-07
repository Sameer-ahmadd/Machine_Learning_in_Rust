use polars::prelude::*;
use xgboost::{parameters, Booster, DMatrix};
pub fn train_xgboost_model(
    x_train: &DataFrame,
    y_train: &DataFrame,
    x_test: &DataFrame,
    y_test: &DataFrame,
) -> anyhow::Result<String> {
    // Transform Polars DataFrames into 2D arrays in row-major order
    let x_train_array = x_train.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_train_array = y_train.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let x_test_array = x_test.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_test_array = y_test.to_ndarray::<Float32Type>(IndexOrder::C)?;

    println!("x_train_array: {:?}", x_train_array);
    println!("x_train_slice: {:?}", x_train_array.as_slice().clone());

    // Convert the 2D arrays into slices &[f32]
    let x_train_slice = x_train_array
        .as_slice()
        .expect("Failed to convert x_train_array to slice - array may not be contiguous");
    let y_train_slice = y_train_array
        .as_slice()
        .expect("Failed to convert y_train_array to slice - array may not be contiguous");
    let x_test_slice = x_test_array
        .as_slice()
        .expect("Failed to convert x_test_array to slice - array may not be contiguous");
    let y_test_slice = y_test_array
        .as_slice()
        .expect("Failed to convert y_test_array to slice - array may not be contiguous");

    // Transform the given DataFrames into XGBoost DMatrix objects
    // for the training set
    let mut dmatrix_train = DMatrix::from_dense(x_train_slice, x_train.height())?;
    dmatrix_train.set_labels(y_train_slice)?;

    // for the testing set
    let mut dmatrix_test = DMatrix::from_dense(x_test_slice, x_test.height())?;
    dmatrix_test.set_labels(y_test_slice)?;

    // train is used to fit parameters, and test is used to evaluate the model
    let evaluation_sets = &[(&dmatrix_train, "train"), (&dmatrix_test, "test")];

    // Set the configuration for training the XGBoost model
    // I guess that here you can set the hyperparameters of the model
    // Challenge: try to find the best hyperparameters for this model
    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dmatrix_train)
        .evaluation_sets(Some(evaluation_sets))
        // .custom_objective_fn(Objective::RegLinear)
        // .custom_evaluation_fn(parameters::EvaluationMetric::RMSE)
        .build()
        .unwrap();

    // Train model
    let model = Booster::train(&training_params).unwrap();

    // Evaluate the model on the test set
    // TODO: investigate what error metric is used by default
    println!("Test {:?}", model.predict(&dmatrix_test).unwrap());

    // Save the model to a file
    let model_path = "boston_housing_model.bin";
    model.save(model_path)?;
    println!("Model saved to {}", model_path);

    Ok(model_path.to_string())
}
