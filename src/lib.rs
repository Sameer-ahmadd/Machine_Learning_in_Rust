// importing all the common functions we used along our project
use anyhow::Ok;
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::Client;
use polars::{prelude::*, series};
use std::vec;
use xgboost::{parameters, Booster, DMatrix};

// Download the CSV file to the disk
pub fn download_csv_file() -> anyhow::Result<String> {
    let url: &str = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";

    // GET the response from the URL
    let response = reqwest::blocking::get(url)?;

    // GET the bytes from the response into the memory
    let bytes = response.bytes()?;

    let file_path = "boston_housing.csv";

    // Write the bytes to the disk
    std::fs::write(file_path, bytes)?;

    Ok(file_path.to_string())
}

// Loaded a CSV file from disk into polars dataframe

pub fn load_csv_file(file_path: &str) -> anyhow::Result<DataFrame> {
    let df = CsvReader::from_path(file_path)?.finish()?;

    println!("Loaded {} rows and {} columns", df.height(), df.width());
    println!("{:?}", df.head(Some(5)));

    Ok(df)
}

// Spliting the Data into training and testing sets
use rand::seq::SliceRandom;
use rand::Rng;

pub fn train_test_split(
    df: &DataFrame,
    test_size_prec: f64,
) -> anyhow::Result<(DataFrame, DataFrame)> {
    // Generate a number of vectors from 0 to numbers of rows.
    let mut indices: Vec<usize> = (0..df.height()).collect();

    // Generate a random number

    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);

    let split_index = (df.height() as f64 * (1.0 - test_size_prec)).ceil() as usize;

    // Split into training and testing indices..

    let train_indices = indices[0..split_index].to_vec();
    let test_indices = indices[split_index..].to_vec();

    // Convert from Vec<usize> to ChunkedArray<Int32Type>
    // We do this transformation because the DataFrame::take method
    // expects a ChunkedArray<Int32Type> as an argument.
    let train_indices_ca =
        UInt32Chunked::from_vec("", train_indices.iter().map(|&x| x as u32).collect());
    let test_indices_ca =
        UInt32Chunked::from_vec("", test_indices.iter().map(|&x| x as u32).collect());

    // Split the indices to train and test dataframes using take() method.
    let train_df = df.take(&train_indices_ca)?;
    let test_df = df.take(&test_indices_ca)?;

    println!("Training set size: {}", train_df.height());
    println!("Testing set size: {}", test_df.height());

    Ok((train_df, test_df))
}

// Spliting the data into features and targets..

pub fn split_features_and_target(df: &DataFrame) -> anyhow::Result<(DataFrame, DataFrame)> {
    // features names
    let features_names = vec![
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b",
        "lstat",
    ];

    // target names

    let target_name = vec!["medv"];

    let features = df.select(&features_names)?;
    let target = df.select(&target_name)?;

    Ok((features, target))
}
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

// Pushes the model to S3 Bucket.

pub async fn pushes_model_to_s3(path_to_model: &str) -> anyhow::Result<()> {
    let region_provider = RegionProviderChain::default_provider().or_else("us-west-2");
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;

    // Create an S3 client so i can talk to S3.

    let client = Client::new(&config);

    // Load the model file into memory
    let model_file_bytes = std::fs::read(path_to_model)?;

    // Upload the model file to the S3 bucket
    // TODO: make this value a parameter to this function
    let bucket_name = "xgboost-rust";
    let key = "boston_housing_model.bin";

    let _result = client
        .put_object()
        .bucket(bucket_name)
        .key(key)
        .body(model_file_bytes.into())
        .send()
        .await?;

    Ok(())
}

/// Download the model from S3 Bucket into memory.
pub async fn download_model_from_s3() -> anyhow::Result<(String)> {
    let region_provider = RegionProviderChain::default_provider().or_else("us-west-2");
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;

    // Create an S3 client so i can talk to S3.

    let client = Client::new(&config);

    let bucket_name = "xgboost-rust";
    let key = "boston_housing_model.bin";

    // First we download the content of the model file from S3 into memory
    let download_path = "downloaded_model.bin";
    let resp = client
        .get_object()
        .bucket(bucket_name)
        .key(key)
        .send()
        .await?;
    let data = resp.body.collect().await?.into_bytes();

    // Save the downloaded bytes to a file on disk
    std::fs::write(download_path, data)?;

    Ok(download_path.to_string())
}
