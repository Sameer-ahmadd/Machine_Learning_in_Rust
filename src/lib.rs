// importing all the common functions we used along our project

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

use polars::prelude::*;

pub fn load_csv_file(file_path: &str) -> anyhow::Result<DataFrame> {
    let df = CsvReader::from_path(file_path)?.finish()?;

    println!("Loaded {} rows and {} columns", df.height(), df.width());
    println!("{:?}", df.head(Some(5)));

    Ok(df)
}
