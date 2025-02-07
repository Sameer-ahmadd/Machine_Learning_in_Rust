use anyhow::Ok;
use clap::Parser;
use house_price_predictor::aws::pushes_model_to_s3;
use house_price_predictor::data::{
    download_csv_file, load_csv_file, split_features_and_target, train_test_split,
};
use house_price_predictor::model::train_xgboost_model;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    bucket_name_s3: String,
    #[arg(short, long)]
    key_s3: String,
}

// Training Script entry point...
// steps
// 1. Download the CSV file to the disk
// 2. Load the CSV file into the memory
// 3. Preprocess the data
// 4. Train the XGBOOST model with this data
// 5. Push the model to the AWS S3 bucket {model registry}

// function to print sum of two numbers
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Starting the training script...");

    // 1. Download the CSV file to the disk
    let csv_file_path = download_csv_file()?;

    // 2. Load the Data from disk into memory

    let df = load_csv_file(&csv_file_path)?;

    //3. Randomly spliting the data into training and testing sets.

    let (train_data, test_data) = train_test_split(&df, 0.2)?;

    // 4. Splitting Data into features and targets.
    let (x_train, y_train) = split_features_and_target(&train_data)?;
    let (x_test, y_test) = split_features_and_target(&test_data)?;

    // 5. Training a xgboost model.
    let path_to_model = train_xgboost_model(&x_train, &y_train, &x_test, &y_test)?;

    // 6. Pushes the model to S3 bucket.

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(pushes_model_to_s3(
        &path_to_model,
        &args.bucket_name_s3,
        &args.key_s3,
    ))?;
    println!("Pushes Model to S3 Bucket.");

    Ok(())
}
