use house_price_predictor::{download_csv_file, load_csv_file, train_test_split};

// Training Script entry point...
// steps
// 1. Download the CSV file to the disk
// 2. Load the CSV file into the memory
// 3. Preprocess the data
// 4. Train the XGBOOST model with this data
// 5. Push the model to the AWS S3 bucket {model registry}

// function to print sum of two numbers
fn main() -> anyhow::Result<()> {
    println!("Starting the training script...");

    // 1. Download the CSV file to the disk
    let csv_file_path = download_csv_file()?;

    // 2. Load the Data from disk into memory

    let df = load_csv_file(&csv_file_path)?;

    //3. Randomly spliting the data into training and testing sets.

    let (train_data, test_data) = train_test_split(&df, 0.2)?;

    Ok(())
}
