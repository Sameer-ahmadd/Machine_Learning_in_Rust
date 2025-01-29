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

    Ok(())
}

// Download the CSV file to the disk
fn download_csv_file() -> anyhow::Result<String> {
    let url: &str = "https://github.com/selva86/datasets/blob/master/BostonHousing.csv";
    // GET the response from the URL
    let response = reqwest::blocking::get(url)?;

    // GET the bytes from the response into the memory
    let bytes = response.bytes()?;

    let file_path = "boston_housing.csv";

    // Write the bytes to the disk
    std::fs::write(file_path, bytes)?;

    Ok(file_path.to_string())
}
