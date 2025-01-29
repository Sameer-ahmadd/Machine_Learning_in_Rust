use house_price_predictor::{download_csv_file, load_csv_file};

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

    Ok(())
}
