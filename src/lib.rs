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

use std::vec;

use anyhow::Ok;
use polars::{prelude::*, series};

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
